"""Build a unified, validated opening suite (UCI move-lines) for BOTH the arena gate and the 5%
self-play seed. A1 = on-distribution openings reconstructed from the training log (last N self-play
iters; the log records EVERY ply, unlike the PCR-sparse npz). A2 = a broad community book for
opening-family breadth A1's narrow repertoire misses. Both are Stockfish-filtered to a playable
"slight-edge" band, deduped by resulting FEN, and family-balanced. Output: openings_suite.txt.

Run from repo root:  python3 tools/build_opening_suite.py
Every emitted line is GUARANTEED legal + replayable + non-terminal (validation gate), so the committed
suite is clean regardless of log-parse edge cases.
"""
import os, re, sys, io, collections, urllib.request, zipfile
import chess, chess.engine, chess.pgn

ROOT      = "/home/krish/Chess-AI_ML-System"
LOG       = f"{ROOT}/chess_ai/training_log.txt"
OUT       = f"{ROOT}/chess_ai/game_engine/openings_suite.txt"
SF        = "/usr/games/stockfish"

TARGET_PLY   = 8          # opening depth in plies (4 moves/side) — matches the old book's depth range
LAST_ITERS   = 3          # A1: reconstruct from the last N self-play iterations in the log
BAND_CP      = 70         # keep |white-POV eval| <= 0.70 pawns (drop already-won/lost openings)
SF_NODES     = 50000      # SF budget per opening (fast, ample for opening eval)
TARGET_SIZE  = 200        # cap on the final suite
A1_QUOTA_FRAC = 0.0       # fraction of the suite from on-distribution A1 (0 = A1 disabled; see note)
A2_URL       = "https://github.com/official-stockfish/books/raw/master/8moves_v3.pgn.zip"

MV_RE = re.compile(r"\[Worker (\d+)\] Move (\d+): ([a-h][1-8][a-h][1-8][qrbn]?)")

# Major opening families for the breadth check (coarse: first 1-2 plies → family tag).
def family_tag(ucis):
    """Coarse opening-family signature from the first plies (for gap-fill / breadth coverage)."""
    b = chess.Board(); sig = []
    for u in ucis[:4]:
        try:
            b.push_uci(u)
        except Exception:
            break
        sig.append(u)
    return " ".join(sig[:3])   # first 3 plies = structure signature


# ─────────────────────────────────────────────────────────────────────────────
def selfplay_sections(lines):
    """Return [(iter, start_idx, end_idx)] for each self-play phase, in order."""
    starts, ends = [], []
    start_re = re.compile(r"=== ITERATION (\d+): SELF-PLAY PHASE")
    end_re   = re.compile(r"✅ ITERATION (\d+) - PHASE 1 COMPLETE")
    for i, ln in enumerate(lines):
        ms = start_re.search(ln)
        if ms: starts.append((int(ms.group(1)), i))
        me = end_re.search(ln)
        if me: ends.append((int(me.group(1)), i))
    secs = []
    for it, si in starts:
        ei = next((e for (eit, e) in ends if eit == it and e > si), None)
        if ei is not None:
            secs.append((it, si, ei))
    return secs


def reconstruct_a1(lines):
    """On-distribution opening lines (first TARGET_PLY plies) from the last LAST_ITERS self-play
    sections. On-distribution = game's first logged move is 'Move 1' (book games start at Move >1)."""
    secs = selfplay_sections(lines)
    secs = secs[-LAST_ITERS:]
    print(f"[A1] self-play sections used: {[it for it, _, _ in secs]}")
    openings = set()
    for _, si, ei in secs:
        cur, moves = {}, {}
        for ln in lines[si:ei]:
            m = MV_RE.search(ln)
            if not m:
                continue
            w, n, u = int(m.group(1)), int(m.group(2)), m.group(3)
            if n == 1:                       # new on-distribution game for this worker
                cur[w] = []; moves[w] = cur[w]
            if w in moves and len(moves[w]) == n - 1:   # contiguous
                moves[w].append(u)
                if len(moves[w]) == TARGET_PLY:
                    openings.add(tuple(moves[w]))
    print(f"[A1] unique {TARGET_PLY}-ply on-distribution openings: {len(openings)}")
    return openings


def fetch_a2():
    """Broad community book (8moves_v3), truncated to TARGET_PLY. Returns set of opening tuples.
    On any failure returns empty set — A2 is gap-fill, A1 + the curated backstop still carry breadth."""
    try:
        print(f"[A2] fetching {A2_URL} ...")
        data = urllib.request.urlopen(A2_URL, timeout=30).read()
        zf = zipfile.ZipFile(io.BytesIO(data))
        pgn_name = next(n for n in zf.namelist() if n.endswith(".pgn"))
        text = zf.read(pgn_name).decode("utf-8", "replace")
        openings = set()
        pgn = io.StringIO(text)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            b = game.board(); ucis = []
            for mv in game.mainline_moves():
                ucis.append(b.uci(mv)); b.push(mv)
                if len(ucis) == TARGET_PLY:
                    break
            if len(ucis) == TARGET_PLY:
                openings.add(tuple(ucis))
        print(f"[A2] unique {TARGET_PLY}-ply community openings: {len(openings)}")
        return openings
    except Exception as e:
        print(f"[A2] fetch/parse failed ({e}) — using curated backstop only.")
        return set()


# Curated broad-family backstop (guarantees breadth even if A2 fetch fails). UCI, >= TARGET_PLY plies.
CURATED = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",   # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6",   # Italian
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",   # Sicilian Open
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",   # Sicilian Accelerated Dragon
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7",   # Sicilian Closed
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7",   # French Classical
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6",   # French Advance
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5",   # Caro-Kann Classical
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6",   # Caro-Kann Advance
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6",   # Scandinavian
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7",   # Pirc Austrian
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4 g8f6",   # Modern
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6",   # Scotch
    "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5 h2h4 g5g4",   # King's Gambit
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4",   # Petrov
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7",   # QGD
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4",   # Slav
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6",   # QGA
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",   # King's Indian
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8",   # Nimzo-Indian
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7",   # Queen's Indian
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5",   # Grünfeld
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7",   # Catalan
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6 g1f3 f8e7",   # Dutch
    "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6 g1f3 f8d6",   # London
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5",   # English Reversed Sicilian
    "c2c4 g8f6 b1c3 e7e6 g1f3 d7d5 d2d4 f8e7",   # English → QGD
    "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6 f1g2 f8e7",   # Réti
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 d2d4 e8g8",   # KID via Nf3
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5 f8b4",   # Four Knights
]


def main():
    # A1 (on-distribution log openings) is disabled: τ=1 self-play makes them exploration noise, not
    # sound openings (see A1_QUOTA note below). Skip the 200MB log read entirely unless re-enabled.
    if A1_QUOTA_FRAC > 0:
        print("Reading log ...")
        with open(LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print(f"log lines: {len(lines):,}")
        a1 = reconstruct_a1(lines)
    else:
        a1 = set()
        print("[A1] disabled — τ=1 self-play openings are exploration noise, not sound openings.")
    a2 = fetch_a2()
    curated = set()
    for line in CURATED:
        t = tuple(line.split()[:TARGET_PLY])
        if len(t) == TARGET_PLY:
            curated.add(t)

    # Candidate pool: A1 primary, then A2 + curated for breadth.
    print(f"\npool: A1={len(a1)} A2={len(a2)} curated={len(curated)}")

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    def validate_and_eval(ucis):
        """Replay legally → no tempo-waste round-trip → non-terminal → SF white-POV eval in band.
        Returns (final_fen, cp) or None. Tempo-waste = a NON-capture move later exactly reversed
        (g1f3…f3g1, the τ=1 junk); capture-then-retreat is allowed (it gained material)."""
        b = chess.Board(); hist = []
        for u in ucis:
            try:
                mv = chess.Move.from_uci(u)
            except Exception:
                return None
            if mv not in b.legal_moves:
                return None
            cap = b.is_capture(mv); f, t = u[:2], u[2:4]
            if any(f == pt and t == pf and not pcap for (pf, pt, pcap) in hist):
                return None                      # non-capture round-trip → tempo-waste junk
            hist.append((f, t, cap)); b.push(mv)
        if b.is_game_over():
            return None
        info = eng.analyse(b, chess.engine.Limit(nodes=SF_NODES))
        cp = info["score"].white().score(mate_score=100000)
        if cp is None or abs(cp) > BAND_CP:
            return None
        return (b.fen(), cp)

    # Build suite (family-balanced, shuffled — NOT alphabetical, which biases to a/b-file moves):
    #   Phase 1: A1 on-distribution, family-capped, up to A1_QUOTA (primary).
    #   Phase 2: curated — guarantee the major-family floor.
    #   Phase 3: A2 community — fill the remaining slots, signatures ABSENT from A1 first (breadth).
    import random as _random
    rng = _random.Random(42)
    a1_list = list(a1); rng.shuffle(a1_list)
    a2_list = list(a2); rng.shuffle(a2_list)

    # A1 DROPPED (A1_QUOTA_FRAC=0): self-play opens with temperature (τ=1, first 16 plies), so the "on-
    # distribution" log openings are EXPLORATION NOISE (knights returning home, random flank pawns),
    # not a sound repertoire — and the balance filter can't catch them (junk-vs-junk is balanced). This
    # net has no broad+sound on-distribution opening set to harvest (narrow greedy repertoire). Goal-2
    # ("solid broad openings, play its strongest game") requires SOUND lines → use the real-games book
    # (A2 = 8moves_v3) + the curated family floor. Raise A1_QUOTA_FRAC only with a per-move soundness gate.
    A1_QUOTA = int(TARGET_SIZE * A1_QUOTA_FRAC)
    suite, seen_fen, fam_count = [], set(), collections.Counter()
    def consider(ucis, source, cap_per_family, quota=None):
        if len(suite) >= TARGET_SIZE or (quota is not None and quota[0] <= 0):
            return False
        fam = family_tag(list(ucis))
        if fam_count[fam] >= cap_per_family:
            return False
        res = validate_and_eval(ucis)
        if res is None:
            return False
        fen, cp = res
        if fen in seen_fen:
            return False
        seen_fen.add(fen); fam_count[fam] += 1
        suite.append((" ".join(ucis), source, fam, cp))
        if quota is not None:
            quota[0] -= 1
        return True

    q = [A1_QUOTA]
    n_a1 = sum(consider(u, "A1", cap_per_family=3, quota=q) for u in a1_list)   # primary, family-spread
    n_cu = sum(consider(u, "curated", cap_per_family=2) for u in sorted(curated))  # family floor
    # A2: absent families first (False < True) → fills the breadth gaps A1's narrow repertoire misses.
    a2_sorted = sorted(a2_list, key=lambda u: family_tag(list(u)) in fam_count)
    n_a2 = sum(consider(u, "A2", cap_per_family=2) for u in a2_sorted)          # breadth to TARGET_SIZE
    eng.quit()

    print(f"\nkept: A1={n_a1} A2={n_a2} curated={n_cu}  total={len(suite)}")
    print(f"distinct family signatures: {len(fam_count)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Unified opening suite (UCI move-lines) — arena gate + 5% self-play seed.\n")
        f.write(f"# Generated by tools/build_opening_suite.py | {TARGET_PLY} plies | "
                f"|eval|<={BAND_CP}cp | {len(suite)} lines | families={len(fam_count)}\n")
        for line, source, fam, cp in suite:
            f.write(line + "\n")
    print(f"\nwrote {len(suite)} openings → {OUT}")
    if len(suite) < TARGET_SIZE // 2:
        print(f"⚠️ only {len(suite)} lines (< {TARGET_SIZE//2}); gate CI will be wider than target.")


if __name__ == "__main__":
    main()
