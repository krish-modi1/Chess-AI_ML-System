#!/usr/bin/env bash
# hyperparams.env.sh — shared AlphaZero run-config for GCP and local launchers.
#
# CALLER MUST SET before sourcing:
#   NCPU      — logical CPU count (e.g., from nproc)
#   VRAM_MIB  — GPU VRAM in MiB (from nvidia-smi)
#
# Exports the full env-var contract consumed by game_engine/main.py.

# ── CRITICAL: pin every process to ONE CPU thread ─────────────────────────────────────────────
# We parallelise across NUM_WORKERS *processes*, not threads. But numpy/MKL/OpenBLAS/torch each
# spin up a thread pool sized to the box (≈16–32) PER process, and idle OpenMP threads spin-wait
# (busy-loop) → 64 workers × ~16 threads ≈ 1000 threads thrashing 32 vCPUs (observed load avg
# ~570, per-worker 11 sim/s vs ~45 with 1 thread). Pinning to 1 thread each removes the thrash;
# the GPU then gets fed and self-play throughput multiplies. Set BEFORE python imports torch.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE   # if any OMP region remains, idle threads sleep instead of spinning

# Heavily oversubscribed: with the small WORKER_BATCH_SIZE below, each search does ~100
# inference round-trips per move, so a worker spends most of its time BLOCKED waiting on the
# GPU. Oversubscribing keeps the CPU busy (ready workers run while others wait) AND feeds the
# GPU a larger pooled batch (NUM_WORKERS×WBS). Target machine: g2-standard-32 = 16 PHYSICAL
# cores / 32 logical (HT), 126 GB RAM, 22 GB L4.
#   - The C++ MCTS traversal releases the GIL, so ~16 workers can compute in parallel; the rest
#     wait on inference. At a ~20-25% CPU duty cycle that balances ~64 workers against 16 cores.
#   - RAM is not the limit: 64 × ~0.4 GB ≈ 25 GB of 126 GB (no per-worker CUDA context; the
#     single server process owns the GPU).
# 60 workers on 30 cores (2 per core), with RESERVED_CORES=2 keeping the GPU-feeding inference
# server + OS on the top 2 cores (dedicated, no worker contention). A starved server can fall
# behind, trip its deadlock timeout, self-kill, and hang every worker — reserving cores prevents
# that cascade. Tune via load average + nvidia-smi.
export NUM_WORKERS=60
export RESERVED_CORES=2

# Per-worker MCTS leaf batch = leaves submitted to the inference server per search iteration.
# num_iterations = SIMULATIONS / WORKER_BATCH_SIZE. 8 → num_iter=100 = full AlphaZero-quality
# sequential search. The 100 inference round-trips/move used to starve the GPU (~13% util) when
# tensors were pickled through mp.Queue, but the SHARED-MEMORY transport (SHM_TRANSPORT, default
# on) moves the bulk tensors into shared buffers and sends only tiny (worker_id, N) signals — so
# the round-trips are cheap and we keep top search quality AND high GPU utilisation.
export WORKER_BATCH_SIZE=8
# Shared-memory inference transport (set 0 to fall back to the legacy pickle-through-queue path).
export SHM_TRANSPORT=1

# VRAM tier → CUDA stream count + a ceiling on the pooled inference batch.
if   (( VRAM_MIB >= 35000 )); then CUDA_STREAMS=8; VRAM_CAP=24000
elif (( VRAM_MIB >= 20000 )); then CUDA_STREAMS=6; VRAM_CAP=16000
elif (( VRAM_MIB >=  8000 )); then CUDA_STREAMS=4; VRAM_CAP=6000
else                               CUDA_STREAMS=2; VRAM_CAP=1536
fi
export CUDA_STREAMS

# Pooled inference batch = one round from every worker (NUM_WORKERS × WORKER_BATCH_SIZE),
# capped by available VRAM.
CUDA_BATCH_SIZE=$(( NUM_WORKERS * WORKER_BATCH_SIZE ))
(( CUDA_BATCH_SIZE > VRAM_CAP )) && CUDA_BATCH_SIZE=$VRAM_CAP
export CUDA_BATCH_SIZE

# FIX: 20 ms partial-batch flush (overrides the 1 s hardcoded default in main.py).
export CUDA_TIMEOUT_INFERENCE=0.02

# ── Watchdog timeouts ─────────────────────────────────────────────────────────
# Sized for the small-WORKER_BATCH_SIZE search (~100 inference round-trips/move → longer
# moves, games, and iterations than the old 1-round-trip config).
export ITERATION_TIMEOUT=172800        # 48h per-iteration deadlock guard (main.py SIGALRM)
export INFERENCE_TIMEOUT_MS=300000     # 5min max wait for ONE inference round-trip (per call)
export SERVER_DEADLOCK_TIMEOUT=1800    # 30min: server self-kills if it processes no batch

# Training DataLoader workers. The self-play server is dead during training so all 32 vCPUs are
# free; 16 (= physical cores) feeds the GPU a 2048-batch without starving it, and prefetch_factor=4
# (in trainer.py) keeps 4 batches ready per worker. COW fork shares the read-only f16 dataset →
# no RAM multiplication. Local launchers override this lower (tiny datasets).
export TRAIN_DL_WORKERS=16
export TRAIN_DL_PREFETCH=4   # batches buffered per worker (local launchers set 1)

# Loop / eval / rules — identical on all platforms.
export SIMULATIONS=1600        # 1200→1600 for the 25% FULL (recorded) searches — sharper policy targets
                              # to re-open the search-vs-net gap (search_probe was in "bump sims" regime:
                              # KL 0.6, Δentropy≈0). Only ~8% slower self-play (playout-cap: 75% stay fast).
export EVAL_SIMULATIONS=800   # kept at 800 so arena/Stockfish-Elo stay comparable to the 1524 anchor

# KataGo-style decided-game playout: once |P(win)-P(loss)| >= threshold for N consecutive
# moves, cap sims for the rest of the game (played to completion, no resignation → honest
# labels). Saves compute on dead/decided games so the budget buys more diverse games.
export DECIDED_VALUE_THRESHOLD=0.9
export DECIDED_PATIENCE=5
export REDUCED_SIMULATIONS=100
export GAMES_PER_WORKER=10
export ITERATIONS=1000
export EVAL_WORKERS=25           # arena: 25 workers × 6 games = 150. 6 is EVEN, and game_id=worker*6+gi
export GAMES_PER_EVAL_WORKER=6   # so each worker alternates W,B,W,B,W,B = 3 White + 3 Black — color is
                                 # balanced PER worker (removes white-advantage bias, robust to a dropped worker).
export ARENA_EARLY_STOP=1        # stop the arena once the 150-game gate is mathematically decided
                                 # (reject if 55% unreachable, promote if already clinched) → the
                                 # saved L4 time goes to self-play. See local/plans/arena-early-stop.md
export PROBE_ON_ITER=1           # auto-run probe_all + search_probe before the arena each iter and
                                 # log markers (value_acc_gap, search_kl) to metrics.json for a trend.
                                 # See local/plans/baked-in-probes.md
export STOCKFISH_GAMES=150       # 150 for a tight BayesElo estimate (±~45 vs ±~75 at 40). Runs EVERY
                                # iter (STOCKFISH_EVERY_ITER) so it's a real per-iter cost, but the Elo
                                # trend is the scoreboard and needs low noise to be readable.
export STOCKFISH_WORKERS=25      # 25 workers × 6 games = 150 → 3 White + 3 Black per worker (balanced,
                                # same game_id%2 alternation as the arena worker).
export STOCKFISH_ELO=1320
export STOCKFISH_EVERY_ITER=0    # OFF: measure champion Elo only on promotion (saves Stockfish compute
                                # — budget). With lineage-off the champion is HELD; arena WR is the per-iter signal.

# Gate RAISED back to 0.55 to PROTECT the iter-3 (1524) champion. iters 13-15 degraded 1524→1222 via
# promotions at 0.52-0.54 — a 0.55 gate rejects those cycling-equal candidates and only promotes a
# CLEARLY-better net. With lineage-off, this HOLDS 1524 instead of bleeding it down.
export PROMOTION_WIN_RATE=0.55
export MAX_MOVES_PER_GAME=800
export EVAL_MAX_MOVES_PER_GAME=800

# RL training recipe — gentle fine-tune of the 1800 Lichess-pretrained net. The raw policy prior
# is weak off the human distribution (gives MCTS-chosen moves ~0.7% median mass), so targets
# demand large shifts; 4 epochs × 3e-4 over a tiny early window overwrote the net (1800→~1300).
# See local/plans/anti-forgetting-levers.md.
export TRAIN_MIN_ITER=8        # window starts at iter-8 (the 1200-sim switch) — drop the stale 800-sim
                              # data from iters 1-7. See on-distribution diagnosis.
export TRAIN_FROM_LINEAGE=0   # OFF for the iter-3 rollback: train each candidate from the FROZEN 1524
                              # champion so the promotion gate PROTECTS it. Lineage (continuous) dragged
                              # the champ 1524→1390→1294 on diffuse data — re-enable once on-distribution
                              # data proves the targets are good (per-iter Elo holding/climbing).
export TEMP_MOVES=8           # was 16. Halve the random-opening window to keep self-play on-distribution.
export TRAIN_EPOCHS=2          # 1→2: extract more from the decorrelated buffer (safe now that cap=20
                              # prevents the overfitting that 4 epochs caused at iter-1)
export TRAIN_LR=3e-4           # PLATEAU LEVER (iter-16): back UP from 1e-4. At 1e-4 the policy froze —
                              # it wasn't absorbing the ~0.5-nat MCTS target signal (search probe:
                              # override ~40%, KL(MCTS‖net)~0.5, yet cand‖champ KL only ~0.003). 3e-4
                              # (the code default) = bigger steps toward the MCTS targets. Watch GNorm
                              # + val loss for instability the first iter; revert to 1e-4 if it spikes.
export KL_ANCHOR_BETA=1.0      # KL(reference ‖ candidate) penalty weight; 0 disables. Kept at 1.0 — the
                              # change this round is the REFERENCE, not β, for clean attribution.
# REFERENCE RESET (iter-22): anchor the KL reference to the LIVE champion instead of the stale 1800
# prior. The net is ~2438 — anchoring to 1800 was pulling it back toward beginner play (accuracy rose
# but it couldn't out-diverge the champion to win → no promotions). Anchoring to current strong play
# lets it specialize past 1800. Grounded in RLHF reward-KL frontier (Gao 2023) + reference-reset
# (arXiv 2510.01555). Flip to 0 to revert to the pretrained-1800 anchor.
export KL_ANCHOR_TO_CHAMPION=1
export MIN_TRAIN_ITERS=3       # self-play only until this many iterations of data exist, then train
export DRAW_MAX_POSITIONS=0        # subsumed by MAX_POSITIONS_PER_GAME below
export MAX_POSITIONS_PER_GAME=20   # cap EVERY game to 20 positions (value-target decorrelation, AlphaGo Nature'16).
                                   # Offline A/B sweet spot: value head 1.5× sharper + best calibration (56%), policy intact;
                                   # caps ≤5 over-confident (shared trunk ≠ AlphaGo's separate value net).
export VALUE_LOSS_WEIGHT=1.0       # no-op: A/B showed value-loss weight has zero effect (flattening is trunk/BN-driven)

# Auxiliary-head trunk regularizers (KataGo-style) to sharpen the signal-starved value head.
# Forward-looking targets baked into the .npz: material=final material margin (MSE), plies=plies-
# to-end (MSE), reply=opponent's next move (CE over 4672). Weights are scale-balanced: reply CE
# (~3-5 on a trained net) is far larger than the MSE losses (~0.1-0.3), so reply gets a small
# weight and the MSE heads larger ones — each contributes a comparable, modest amount vs p_loss
# (~1.3) / v_loss (~0.85). CONFIRM/tune from iter-8's epoch-1 "aux (raw, pre-weight)" log line.
# 0 = off. See local/plans/auxiliary-targets.md.
export AUX_W_MAT=1.0       # 0.5→1.0: the value head trains SLOWLY (sparse outcome signal). Material
                          # margin is the most value-correlated aux target, so weight it harder to
                          # regularize the trunk toward value structure → faster value-head learning.
                          # (raw ~0.4 × 1.0 ≈ 10% of p_loss — into the 10-20% target band.)
export AUX_W_PLIES=0      # DROPPED — plies head wasn't learning (flat ~0.14), just trunk-gradient noise.
export AUX_W_REPLY=0.03

# ── Search/training upgrades (master-table easy wins) — see local/plans/upgrades-1-6.md ──
# ENABLED at the recommended values (Lc0/KataGo) from iter-10 onward. Each is env-isolated, so if
# iter-10 regresses, disable individually to find the culprit (forced playouts is the most novel —
# watch it first). The C++ ones (FPU/cpuct/forced) take effect on the .so rebuild run_gcp.sh does.
export FPU_REDUCTION=0.5        # 1) First-Play-Urgency: unvisited child Q = parent_Q − 0.5·√(explored P), 0 at root
                               #    KEPT — it *concentrates* search (less exploration of unvisited), aligns w/ on-distribution.
export CPUCT_FACTOR=1.0        # 3) PULLED BACK 2.0→1.0 (AlphaZero baseline). iter-9(=1) vs iter-10(=2) showed ZERO
                               #    effect on target diffuseness, and 2.0 only adds off-distribution exploration.
export FORCED_PLAYOUT_K=0      # 4) PULLED BACK 2.0→0 (disabled). No effect on diffuseness, most-novel, and it
                               #    spreads visits (against the on-distribution thesis). See selfplay-offdistribution memory.
export FULL_SEARCH_PROB=0.25    # 2) playout-cap: 25% of moves full+recorded, rest fast+unrecorded (Dirichlet off)
export FAST_SIMULATIONS=200     #    fast-search sim count
export SWA_ENABLE=1             # 5) stochastic weight averaging → offline swa_model.pth (probe vs candidate)
export SWA_DECAY=0.75
# 6) aux weights above (AUX_W_*) — tune from the epoch-1 "aux (raw, pre-weight)" log, not a fixed change.
