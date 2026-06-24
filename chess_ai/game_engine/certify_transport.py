#!/usr/bin/env python3
"""
Certify the SHM inference transport (the path self-play / arena / eval all use).

Drives the REAL MCTSWorker._batch_inference_callback through the REAL InferenceServer with many
concurrent workers, and checks every worker gets back the result for ITS OWN input — vs a direct
model() reference AND vs its own previous call (reference-free self-consistency). This is the
regression test for the 2026-06-19 CUDA-stream-sync corruption bug (process_batch copied results
before the side stream finished → ~5%+ wrong leaf evals). Fix = stream.synchronize() before .cpu().

Run from the repo root OR chess_ai/ (paths are derived from __file__, cwd-independent):
    CERT_N=1  CERT_STREAMS=8 python chess_ai/game_engine/certify_transport.py   # validity gate (must PASS)
    CERT_N=12 CERT_STREAMS=8 python chess_ai/game_engine/certify_transport.py   # production concurrency

PASS  → max_err ~1e-3, vs-ref=0, self-inconsistent=0  → transport faithful → data is clean.
FAIL  → self-inconsistent>0 (reference-free) → transport corrupts leaf evals. Needs a GPU.
Validity gate: N=1 (no concurrency) MUST pass — a single worker can't desync; an N=1 failure means
the harness/env is wrong, not the transport.
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))      # .../chess_ai/game_engine
CHESS_AI = os.path.dirname(HERE)                        # .../chess_ai
sys.path.insert(0, CHESS_AI)                            # for `from game_engine...`
sys.path.insert(0, HERE)                               # for bare `import mcts_engine_cpp` (the .so)
from game_engine.neural_net import InferenceServer
from game_engine.cnn import ChessCNN

MODEL  = os.path.join(HERE, "model", "best_model.pth")
N      = int(os.environ.get("CERT_N", "12"))
WB     = 8
CB     = N * WB
ROUNDS = int(os.environ.get("CERT_ROUNDS", "150"))
TOL    = 0.5


def load_state():
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)  # trusted self-ckpt; carries numpy metadata
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def random_board(rng):
    import chess
    b = chess.Board()
    for _ in range(rng.randint(0, 24)):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng.choice(ms))
    return b


def worker_proc(wid, in_q, out_q, shm_inp, shm_pol, shm_val, seed, result_q):
    from game_engine.mcts_worker_cpp import MCTSWorker, _cpp_board_from_history
    torch.set_num_threads(1)
    rng = random.Random(seed)
    worker = MCTSWorker(wid, in_q, out_q, simulations=1, batch_size=WB,
                        shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
    worker._cpp_history = []

    leaves = [_cpp_board_from_history(random_board(rng)) for _ in range(WB)]
    tens = np.array([lf.to_tensor(worker._cpp_history) for lf in leaves])
    model = ChessCNN().eval()
    model.load_state_dict(load_state(), strict=False)
    with torch.no_grad():
        rp, rv = model(torch.from_numpy(tens).float())
    ref_pol = rp.numpy()
    ref_vsc = (torch.softmax(rv, 1)[:, 0] - torch.softmax(rv, 1)[:, 2]).numpy()
    del model

    max_err, bad, done, selfbad = 0.0, 0, 0, 0
    prev = None
    for r in range(ROUNDS):
        try:
            pol, vsc = worker._batch_inference_callback(leaves)
        except Exception as e:
            result_q.put((wid, "ERR", str(e)[:80], done)); return
        e = max(float(np.abs(pol - ref_pol).max()), float(np.abs(vsc - ref_vsc).max()))
        max_err = max(max_err, e); done += 1
        if e > TOL:
            bad += 1
        if prev is not None:
            se = max(float(np.abs(pol - prev[0]).max()), float(np.abs(vsc - prev[1]).max()))
            if se > TOL:
                selfbad += 1
        prev = (pol, vsc)
    result_q.put((wid, max_err, bad, done, selfbad))


def run_server(server):
    server.loop()


def main():
    mp.set_start_method("spawn", force=True)
    streams = int(os.environ.get("CERT_STREAMS", "8"))
    shm_inp = torch.empty((N, WB, 120, 8, 8), dtype=torch.float32).share_memory_()
    shm_pol = torch.empty((N, WB, 4672), dtype=torch.float32).share_memory_()
    shm_val = torch.empty((N, WB, 3), dtype=torch.float32).share_memory_()
    server = InferenceServer(MODEL, batch_size=CB, timeout=0.02, streams=streams,
                             shm_inp=shm_inp, shm_pol=shm_pol, shm_val=shm_val)
    wq = [server.register_worker(i) for i in range(N)]
    sp = mp.Process(target=run_server, args=(server,)); sp.start()
    time.sleep(8)

    result_q = mp.Queue()
    procs = []
    for i in range(N):
        pr = mp.Process(target=worker_proc,
                        args=(i, server.input_queue, wq[i], shm_inp, shm_pol, shm_val,
                              2000 + i, result_q))
        pr.start(); procs.append(pr)
    results = [result_q.get() for _ in range(N)]
    for pr in procs:
        pr.join(timeout=10)
    server.input_queue.put("STOP")
    sp.join(timeout=10)
    if sp.is_alive():
        sp.terminate()

    print("\n" + "=" * 64 + f"   (N={N}, streams={streams})")
    worst, failed, selffail, errs = 0.0, 0, 0, 0
    for r in sorted(results):
        if r[1] in ("ERR", "TIMEOUT"):
            print(f"  worker {r[0]:2d}: ERR {r[2]}"); errs += 1; continue
        wid, max_err, bad, done, selfbad = r
        ok = (bad == 0 and selfbad == 0 and max_err <= TOL)
        print(f"  worker {wid:2d}: {'PASS' if ok else 'FAIL'}  max_err={max_err:.2e}  "
              f"vs-ref={bad}/{done}  self-inconsistent={selfbad}/{done}")
        worst = max(worst, max_err); failed += bad; selffail += selfbad
    print("=" * 64)
    if errs == 0 and failed == 0 and selffail == 0 and worst <= TOL:
        print(f"PASS (N={N}) — transport faithful, {N*ROUNDS} real round-trips, worst_err={worst:.2e}")
    else:
        print(f"FAIL (N={N}) — worst_err={worst:.2e}, vs-ref={failed}, self-inconsistent={selffail}, errs={errs}")


if __name__ == "__main__":
    main()
