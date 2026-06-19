#!/usr/bin/env python3
"""Round-robin Elo on ONE consistent scale, SERVER-MODE (like the training loop).

Each model runs on a SHARED GPU InferenceServer (2 copies max on the GPU at a time); 92 CPU
workers route their MCTS leaves to the server via SHM — so worker count is cheap (no per-process
model copies → no OOM). Plays every model-vs-model pair AND every model-vs-A_low pair, both
colors, emits PGN, then one multi-player BayesElo pinned to A_low = 1320 (anchors.json).

Doubles as a live test of: the inference-server/SHM path, the rebuilt BayesElo, and the anchors.

Run on CARC via carc_eval.slurm (needs hyperparams.env.sh sourced for the server config).
"""
import argparse, collections, datetime, io, itertools, json, os, re, subprocess, sys, time
import multiprocessing as mp
import warnings; warnings.filterwarnings("ignore")
import numpy as np, chess, chess.pgn, chess.engine

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "game_engine"))
from game_engine.chess_env import ChessGame                                  # type: ignore
from game_engine.mcts_worker_cpp import MCTSWorker                            # type: ignore
from game_engine.main import _start_inference_server, _stop_inference_server, setup_child_logging  # type: ignore

# Server config from env (carc_eval.slurm sources hyperparams.env.sh + overrides).
WORKER_BATCH = int(os.environ.get("WORKER_BATCH_SIZE", 8))
CUDA_STREAMS = int(os.environ.get("CUDA_STREAMS", 8))
CUDA_TIMEOUT = float(os.environ.get("CUDA_TIMEOUT_INFERENCE", 0.02))


def _pgn(white, black, result, moves):
    g = chess.pgn.Game()
    g.headers["White"] = white; g.headers["Black"] = black; g.headers["Result"] = result
    g.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    node = g
    for m in moves:
        try:
            node = node.add_variation(chess.Move.from_uci(m))
        except Exception:
            pass
    buf = io.StringIO(); print(g, file=buf, end="\n\n"); return buf.getvalue()


def _mk_worker(wid, srv, sims, seed):
    """MCTSWorker bound to a server: srv=(server, sp, wq, shm)."""
    return MCTSWorker(wid, srv[0].input_queue, srv[2][wid], simulations=sims, batch_size=WORKER_BATCH,
                      seed=seed, shm_inp=srv[3][0], shm_pol=srv[3][1], shm_val=srv[3][2])


def _rr_model_worker(wid, n_games, color_start, sa, sb, name_a, name_b, sims, max_moves, q):
    """Model-vs-model: both sides routed to their own shared server. Emits PGN per game."""
    setup_child_logging()
    import random
    seed = wid + int(time.time()) % 10000; np.random.seed(seed); random.seed(seed)
    wa = _mk_worker(wid, sa, sims, seed); wb = _mk_worker(wid, sb, sims, seed)
    pgns = []
    try:
        for gi in range(n_games):
            a_white = ((color_start + gi) % 2 == 0)
            game = ChessGame(); history = collections.deque(maxlen=7)
            wa.reset_cache(); wb.reset_cache()
            legal = list(game.board.legal_moves)
            if legal:                                   # one random opening ply for variety
                op = random.choice(legal).uci(); history.appendleft(game.board.copy())
                game.push(op); wa.advance_root(op); wb.advance_root(op)
            try:
                while not game.is_over and len(game.moves) < max_moves:
                    a_turn = ((game.board.turn == chess.WHITE) == a_white)
                    mv, _ = (wa if a_turn else wb).search(game, temperature=0.1,
                                                          history=list(history), use_dirichlet=False)
                    if mv is None: break
                    wa.advance_root(mv); wb.advance_root(mv)
                    history.appendleft(game.board.copy()); game.push(mv)
            except Exception as e:
                print(f"[RR W{wid}] {name_a} v {name_b} g{gi}: {e}"); continue
            res = game.result if game.is_over else "1/2-1/2"
            pgns.append(_pgn(name_a if a_white else name_b, name_b if a_white else name_a, res, game.moves))
    except Exception as e:
        print(f"[RR W{wid}] crashed: {e}")
    q.put(pgns)


def _rr_sf_worker(wid, n_games, color_start, sm, model_name, anchor_name, uci_elo, nodes,
                  sims, max_moves, stockfish_path, q):
    """Model-vs-A_low: model on the shared server, A_low = SF (UCI_Elo + fixed nodes) on CPU."""
    setup_child_logging()
    import random
    seed = wid + int(time.time()) % 10000; np.random.seed(seed); random.seed(seed)
    wm = _mk_worker(wid, sm, sims, seed)
    try:
        eng = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            eng.configure({"UCI_LimitStrength": True, "UCI_Elo": int(uci_elo)})
        except Exception as e:
            print(f"[RR W{wid}] anchor UCI_Elo config failed ({e}) — SF FULL STRENGTH")
    except Exception as e:
        print(f"[RR W{wid}] SF launch failed: {e}"); q.put([]); return
    pgns = []
    for gi in range(n_games):
        m_white = ((color_start + gi) % 2 == 0)
        game = ChessGame(); history = collections.deque(maxlen=7); wm.reset_cache()
        try:
            while not game.is_over and len(game.moves) < max_moves:
                if (game.board.turn == chess.WHITE) == m_white:
                    mv, _ = wm.search(game, temperature=0.1, history=list(history), use_dirichlet=False)
                else:
                    mv = eng.play(game.board, chess.engine.Limit(nodes=int(nodes))).move.uci()
                if mv is None: break
                wm.advance_root(mv); history.appendleft(game.board.copy()); game.push(mv)
        except Exception as e:
            print(f"[RR W{wid}] {model_name} v {anchor_name} g{gi}: {e}"); continue
        res = game.result if game.is_over else "1/2-1/2"
        pgns.append(_pgn(model_name if m_white else anchor_name,
                         anchor_name if m_white else model_name, res, game.moves))
    try: eng.quit()
    except Exception: pass
    q.put(pgns)


def _run_pairing(spec_a, spec_b, games, workers, sims, max_moves, stockfish):
    """spec = ('model',name,path) | ('sf',name,uci_elo,nodes). Returns list[pgn_str]."""
    n = max(1, min(games, workers))
    per = [games // n + (1 if i < games % n else 0) for i in range(n)]
    server_batch = n * WORKER_BATCH                     # MUST match worker count or the server stalls
    q = mp.Queue(); procs = []; servers = []; off = 0
    name_a, name_b = spec_a[1], spec_b[1]
    try:
        if spec_a[0] == "model" and spec_b[0] == "model":
            sa = _start_inference_server(spec_a[2], n, WORKER_BATCH, server_batch, CUDA_STREAMS, CUDA_TIMEOUT)
            sb = _start_inference_server(spec_b[2], n, WORKER_BATCH, server_batch, CUDA_STREAMS, CUDA_TIMEOUT)
            servers = [sa, sb]
            for i in range(n):
                if per[i] == 0: continue
                p = mp.Process(target=_rr_model_worker,
                               args=(i, per[i], off, sa, sb, name_a, name_b, sims, max_moves, q))
                p.start(); procs.append(p); off += per[i]
        else:
            mspec = spec_a if spec_a[0] == "model" else spec_b
            sfspec = spec_b if spec_a[0] == "model" else spec_a
            sm = _start_inference_server(mspec[2], n, WORKER_BATCH, server_batch, CUDA_STREAMS, CUDA_TIMEOUT)
            servers = [sm]
            for i in range(n):
                if per[i] == 0: continue
                p = mp.Process(target=_rr_sf_worker,
                               args=(i, per[i], off, sm, mspec[1], sfspec[1], sfspec[2], sfspec[3],
                                     sims, max_moves, stockfish, q))
                p.start(); procs.append(p); off += per[i]
        pgns = []
        for _ in procs:
            try: pgns += q.get(timeout=3600)
            except Exception: pass
        for p in procs:
            p.join(timeout=10)
            if p.is_alive(): p.terminate(); p.join(timeout=5)
    finally:
        for s in servers:
            _stop_inference_server(s[0], s[1])
    return pgns


def _bayeselo_ladder(pgn_path, bayeselo_bin, anchor_name, anchor_elo, cwd):
    pgn_path = os.path.abspath(pgn_path)
    cmds = f"readpgn {pgn_path}\nelo\nmm\nexactdist\nratings\nx\n"
    out = subprocess.run([bayeselo_bin], input=cmds, capture_output=True, text=True, cwd=cwd).stdout
    ratings = {}
    for ln in out.splitlines():
        m = re.match(r"\s*\d+\s+(\S+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)", ln)
        if m:
            ratings[m.group(1)] = {"elo": int(m.group(2)), "ci": (int(m.group(3)) + int(m.group(4))) // 2,
                                   "games": int(m.group(5))}
    if anchor_name in ratings:
        off = anchor_elo - ratings[anchor_name]["elo"]
        for r in ratings.values(): r["elo"] += off
    else:
        print("WARNING: anchor not in BayesElo output — ratings are un-pinned.\n" + out)
    return ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="comma list name:path (models)")
    ap.add_argument("--anchors-json", default="chess_ai/anchors.json")
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish"))
    ap.add_argument("--bayeselo", default=os.path.join(_ROOT, "BayesElo", "bayeselo"))
    ap.add_argument("--games", type=int, default=184)
    ap.add_argument("--workers", type=int, default=92)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--max-moves", type=int, default=300)
    ap.add_argument("--out-dir", default=f"local/rr_{int(time.time())}")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.anchors_json))
    a = cfg["anchors"]["A_low"]
    anchor_spec = ("sf", "A_low", a["uci_elo"], a["nodes"]); anchor_elo = a["elo"]

    models = []
    for tok in args.players.split(","):
        name, path = tok.split(":", 1)
        if not os.path.isabs(path) and not path.startswith("chess_ai"):
            path = os.path.join("chess_ai", path)
        models.append(("model", name, path))

    pairings = list(itertools.combinations(models, 2)) + [(m, anchor_spec) for m in models]
    print(f"Round-robin (server-mode): {len(models)} models + A_low | {len(pairings)} pairings × "
          f"{args.games} games | {args.sims} sims | {args.workers} workers", flush=True)

    combined = os.path.join(args.out_dir, "all_games.pgn")
    with open(combined, "w") as f:
        for sa, sb in pairings:
            print(f"[run] {sa[1]} vs {sb[1]} ...", flush=True)
            pgns = _run_pairing(sa, sb, args.games, args.workers, args.sims, args.max_moves, args.stockfish)
            f.write("".join(pgns)); f.flush()
            print(f"[run]   {len(pgns)} games", flush=True)

    ratings = _bayeselo_ladder(combined, args.bayeselo, "A_low", anchor_elo, _ROOT)
    print(f"\n=== Elo ladder (BayesElo, A_low pinned = {anchor_elo}) ===")
    for name, r in sorted(ratings.items(), key=lambda kv: -kv[1]["elo"]):
        print(f"  {name:<20} {r['elo']:>5} ± {r['ci']:<3}  ({r['games']} games)")
    json.dump(ratings, open(os.path.join(args.out_dir, "ladder.json"), "w"), indent=2)
    print(f"\nPGN: {combined}\nLadder: {os.path.join(args.out_dir, 'ladder.json')}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
