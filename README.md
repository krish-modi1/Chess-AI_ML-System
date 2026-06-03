# Chess AI — PocketZero

AlphaZero-style chess engine trained entirely from self-play. PyTorch ResNet + C++ MCTS.

**Current ELO**: ~1200 (50 self-play iterations). Active training in progress. Target: 1800.

---

## Model Download

```bash
# From chess_ai/
gdown 1FHQQI9hNmIxAZd6zmX6QO8oow5ekjgGs -O game_engine/model/best_model.pth
```

Or manually: https://drive.google.com/file/d/1FHQQI9hNmIxAZd6zmX6QO8oow5ekjgGs

> 120-channel AlphaZero-style ResNet (256ch, 20 blocks). 416 MB.

---

## Training on GCP

```bash
git clone <repo>
cd chess_ai
bash run_gcp.sh           # foreground, logs to stdout
bash run_gcp.sh --background  # nohup, logs to logs/training.log
```

`run_gcp.sh` handles everything: GPU detection, system deps, Python env, C++ MCTS build, model download, hyperparameter tuning by VRAM tier, and launch.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Input | 120 × 8 × 8 float32 (8 history frames × 14 planes + 8 aux) |
| Network | 256-channel ResNet, 20 blocks, SE from block 10, Mish activation |
| Policy head | 8192 logits (→ 4672 AlphaZero encoding, upcoming) |
| Value head | Scalar tanh (→ WDL softmax, upcoming) |
| MCTS | C++ pybind11 extension, 800 sims/move, tree reuse |
| Training | Adam + CosineAnnealingLR + AMP + horizontal-flip augmentation |

---

## Web Interface

Being rebuilt as a hosted app. Planned after hitting ELO milestones.

---

## Project Structure

```
chess_ai/
├── game_engine/          # Core ML + MCTS code
│   ├── main.py           # Training orchestration
│   ├── cnn.py            # Neural network
│   ├── trainer.py        # Training loop
│   ├── evaluation.py     # Arena + Stockfish eval (parallel)
│   ├── chess_env.py      # Board representation + tensor encoding
│   ├── mcts_worker_cpp.py  # C++ MCTS worker wrapper
│   └── src/              # C++ MCTS source (pybind11)
├── data/self_play/       # Generated training data (.npz)
├── run_gcp.sh            # GCP setup + training launcher
└── smoke_test.sh         # 132-check system verification
```
