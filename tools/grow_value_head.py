"""Migrate a checkpoint to the global-pooling value head (cnn.py ValueHead) and WARM-TRAIN it.

WHY: cnn.py replaced the old 1-channel-bottleneck value head (Conv 320→1 → 64 cells) with a
KataGo/Lc0-style global avg+max pooling head (`value_head_gp`). The change is NOT function-preserving
(unlike the Net2Net widen), so loading an old champion into the new arch leaves the value head at
random init. A random value head lobotomizes the champion → self-play plays weak AND the arena gate
becomes meaningless (a trained candidate would falsely crush the lobotomized champion). So we warm-train
ONLY the new value head, with the TRUNK + POLICY FROZEN (policy stays byte-identical → the champion's
move selection is preserved), until value accuracy recovers. Then the pipeline resumes normally.

Runs on the box (needs the self-play .npz window + a GPU). Usage from repo root:
  python3 tools/grow_value_head.py \
      --ckpt chess_ai/game_engine/model/best_model.pth \
      --data-glob 'chess_ai/game_engine/model/iter_*' \
      --epochs 3 --max-pos 300000
Backs up the original to <ckpt>.prevarch.bak and resets lineage (candidate/swa → .bak). Idempotent-safe:
if the ckpt already has value_head_gp keys it just re-warms.
"""
import argparse, glob, os, re, shutil, sys, time
import numpy as np
import torch
import torch.nn as nn

# Import the REAL (new-arch) ChessCNN so this tool never drifts from cnn.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "chess_ai", "game_engine"))
from cnn import ChessCNN  # noqa: E402


def load_state(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = raw.get("model_state_dict", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def iter_dirs_newest_first(data_glob):
    dirs = [d for d in glob.glob(data_glob) if os.path.isdir(d)]
    def it_num(d):
        m = re.search(r"iter_?(\d+)", os.path.basename(d))
        return int(m.group(1)) if m else -1
    return sorted(dirs, key=it_num, reverse=True)


def load_window(data_glob, max_pos, device):
    """Gather up to max_pos (states, value-class) pairs, newest iterations first.
    States kept as float16 on CPU (cast to float32 per-batch); values as int64 class ids."""
    states, values, total = [], [], 0
    for d in iter_dirs_newest_first(data_glob):
        for f in glob.glob(os.path.join(d, "*.npz")):
            try:
                with np.load(f) as z:
                    s = z["states"]; v = z["values"]
            except Exception:
                continue
            if len(s) == 0:
                continue
            states.append(np.asarray(s, dtype=np.float16))
            values.append(np.asarray(v, dtype=np.int64))
            total += len(s)
        if total >= max_pos:
            break
    if not states:
        raise SystemExit(f"No .npz found under {data_glob} — run this on the box where self-play data lives.")
    S = torch.from_numpy(np.concatenate(states)[:max_pos])          # (N,120,8,8) f16, CPU
    V = torch.from_numpy(np.concatenate(values)[:max_pos])          # (N,)        i64, CPU
    return S, V


def warm_train(model, S, V, device, epochs, batch, lr):
    """Train ONLY value_head_gp; trunk+policy+aux frozen (trunk in eval → BN stats fixed)."""
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("value_head_gp"))
    model.eval()                       # freeze trunk BN running stats
    model.value_head_gp.train()        # let the new head's BN update
    opt = torch.optim.Adam(model.value_head_gp.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    N = len(S)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot, correct, seen, t0 = 0.0, 0, 0, time.time()
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            x = S[idx].to(device).float()
            y = V[idx].to(device)
            _, v = model(x)            # inference path: (policy, value_logits)
            loss = ce(v, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx); seen += len(idx)
            correct += (v.argmax(1) == y).sum().item()
        print(f"  [warm] epoch {ep+1}/{epochs}  value_CE={tot/seen:.4f}  value_acc={100*correct/seen:.1f}%  ({time.time()-t0:.0f}s)")


def backup(path, suffix=".prevarch.bak"):
    if os.path.exists(path):
        bak = path + suffix
        shutil.copyfile(path, bak)
        return bak
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="chess_ai/game_engine/model/best_model.pth")
    ap.add_argument("--data-glob", default="chess_ai/game_engine/model/iter_*")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-pos", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-warmup", action="store_true", help="migrate arch only, leave value head at init")
    ap.add_argument("--keep-lineage", action="store_true", help="do NOT reset candidate/swa")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Migrate arch: load old state into the new-arch net (strict=False → value head fresh).
    sd = load_state(args.ckpt, device)
    model = ChessCNN().to(device)
    ld = model.load_state_dict(sd, strict=False)
    nonvalue_missing = [k for k in ld.missing_keys if not k.startswith("value_head_gp")]
    print(f"  migrate: {len(ld.missing_keys)} missing "
          f"(value_head_gp fresh, non-value missing={nonvalue_missing}), "
          f"{len(ld.unexpected_keys)} unexpected (old value_head.* dropped)")
    assert not nonvalue_missing, f"trunk/policy/aux did NOT carry over: {nonvalue_missing}"

    # 2. Warm-train the new value head (trunk frozen) so the champion keeps its strength.
    if not args.no_warmup:
        print(f"Loading warm-train window (≤{args.max_pos} pos) from {args.data_glob} ...")
        S, V = load_window(args.data_glob, args.max_pos, device)
        print(f"  loaded {len(S):,} positions "
              f"(WDL mix: {np.bincount(V.numpy(), minlength=3).tolist()})")
        warm_train(model, S, V, device, args.epochs, args.batch, args.lr)
    else:
        print("  --no-warmup: value head left at random init (champion will be weak until retrained).")

    # 3. Back up original, write migrated checkpoint (drop optimizer/scheduler → fresh across arch change).
    bak = backup(args.ckpt)
    print(f"  backed up original → {bak}")
    torch.save({"model_state_dict": model.state_dict()}, args.ckpt)
    print(f"  ✓ wrote migrated global-pooling checkpoint → {args.ckpt}")

    # 4. Reset lineage: candidate/swa are stale-arch; delete (with backup) so they rebuild from the new champion.
    if not args.keep_lineage:
        mdir = os.path.dirname(args.ckpt)
        for name in ("candidate.pth", "swa_model.pth"):
            p = os.path.join(mdir, name)
            if os.path.exists(p):
                b = backup(p)
                os.remove(p)
                print(f"  reset lineage: {name} → {b} (removed)")


if __name__ == "__main__":
    main()
