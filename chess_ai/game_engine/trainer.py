import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import gc
import os
import glob
import sys
import time

# Ensure we can import from the parent directory
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN

def _build_h_flip_perm():
    """
    Policy index permutation for horizontal (left-right) board flip.
    AlphaZero 4672 encoding: 64 src squares × 73 planes.
    Mirroring reverses file direction, which maps:
      Queen dirs: N↔N, NE↔NW, E↔W, SE↔SW, S↔S (same distance)
      Knight dirs: each L-shape mirrors its file delta
      Underpromotion dirs: left(df<0) ↔ right(df>0)
    """
    perm = np.arange(4672, dtype=np.int64)
    # Queen direction flip table (dir → mirrored dir)
    dir_flip = [0, 7, 6, 5, 4, 3, 2, 1]
    # Knight direction flip: (df,dr) → (-df,dr)
    # 0=(+1,+2)→7=(-1,+2), 1=(+2,+1)→6=(-2,+1), 2=(+2,-1)→5=(-2,-1), 3=(+1,-2)→4=(-1,-2)
    knight_flip = [7, 6, 5, 4, 3, 2, 1, 0]
    # Underpromotion dir flip: NW(0,df=-1) ↔ NE(2,df=+1), N(1) stays
    underprom_dir_flip = [2, 1, 0]

    for src in range(64):
        src_file = src % 8
        src_rank = src // 8
        new_src = (7 - src_file) + src_rank * 8

        for plane in range(73):
            if plane < 56:  # Queen-like move
                orig_dir = plane // 7
                dist     = plane % 7
                new_plane = dir_flip[orig_dir] * 7 + dist
            elif plane < 64:  # Knight move
                new_plane = 56 + knight_flip[plane - 56]
            else:  # Underpromotion
                idx = plane - 64
                udir, piece = idx // 3, idx % 3
                new_plane = 64 + underprom_dir_flip[udir] * 3 + piece

            perm[new_src * 73 + new_plane] = src * 73 + plane

    return torch.from_numpy(perm).long()

_H_FLIP_PERM = _build_h_flip_perm()

# Load-time position subsampling (the .npz on disk is untouched). Curbs value-overfitting from
# correlated within-game positions (AlphaGo, Nature 2016). 0 = disabled.
#   DRAW_MAX_POSITIONS     — stricter cap for DRAW games only (rebalances draw labels).
#   MAX_POSITIONS_PER_GAME — cap for EVERY game (decorrelation); applied as min() with the draw cap.
DRAW_MAX_POSITIONS     = int(os.environ.get("DRAW_MAX_POSITIONS", 0))
MAX_POSITIONS_PER_GAME = int(os.environ.get("MAX_POSITIONS_PER_GAME", 0))
# Weight on the value loss in the combined objective (AlphaZero uses a low-scale value term;
# "Policy or Value?" CoG 2019 shows the equal p+v sum is suboptimal). 1.0 = current behavior.
VALUE_LOSS_WEIGHT = float(os.environ.get("VALUE_LOSS_WEIGHT", 1.0))

def scan_window_files(data_dir, window_size=20):
    """Scan the last `window_size` iteration folders and return [(path, n_positions)] for every
    valid .npz, NEWEST-iteration-first (so a downstream cap/chunk keeps the most recent data).
    Shape-validates each file (states (N,120,8,8), policies (N,4672)); skips corrupt/empty."""
    if not os.path.exists(data_dir):
        print(f"Warning: No data found in {data_dir}")
        return []
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("iter_")]
    try:
        sorted_subdirs = sorted(subdirs, key=lambda x: int(x.split("_")[1]))
    except ValueError:
        sorted_subdirs = []
    active_folders = sorted_subdirs[-window_size:] if sorted_subdirs else []
    if not active_folders:
        print("No iteration data folders found.")
        return []
    print(f"Data Window: last {len(active_folders)} iterations: {active_folders[0]} to {active_folders[-1]}")
    all_files = []
    for folder in reversed(active_folders):            # newest iteration first
        all_files.extend(glob.glob(os.path.join(data_dir, folder, "*.npz")))
    plan = []
    for f in all_files:
        try:
            with np.load(f, allow_pickle=True) as d:
                s_shape = d['states'].shape
                p_shape = d['policies'].shape
            n = s_shape[0]
            if (len(s_shape) != 4 or s_shape[1:] != (120, 8, 8) or
                    len(p_shape) != 2 or p_shape[1] != 4672 or n == 0):
                print(f"Skipping corrupt file {f}: s={s_shape} p={p_shape}")
                continue
            plan.append((f, n))
        except Exception as e:
            print(f"Error scanning {f}: {e}")
    return plan


class ChessDataset(Dataset):
    def __init__(self, data_dir, window_size=20, max_positions=None, file_plan=None):
        # Two load modes:
        #  • file_plan=None (default): scan the last `window_size` iters newest-first and cap at
        #    max_positions (MAX_TRAIN_POSITIONS — a RAM ceiling, ~24.7 KB/position in f16). Used
        #    when the whole training set is loaded in one shot.
        #  • file_plan=[(path, n_take), …]: load EXACTLY those files (one RAM chunk), no cap. The
        #    chunked trainer uses this to sweep a window larger than RAM across several loads.
        self.total_positions = 0
        self.num_games = 0
        self._game_ids = np.empty(0, dtype=np.int32)

        if file_plan is None:
            if max_positions is None:
                max_positions = int(os.environ.get("MAX_TRAIN_POSITIONS", 4_500_000))
            scan = scan_window_files(data_dir, window_size)
            load_plan, remaining = [], max_positions
            for f, n in scan:
                if remaining <= 0:
                    break
                take = min(n, remaining)
                load_plan.append((f, take))
                remaining -= take
        else:
            load_plan = file_plan

        total = sum(t for _, t in load_plan)
        if total == 0:
            return

        # Pre-allocate float16 arrays: states/policies are 0-1 range so f16 is lossless
        # for states (0/1 integers) and sufficient for policies (~4 sig figs).
        # .float() in __getitem__ casts back to float32 for the model.
        # Peak RAM = these arrays (~49 GB at 2M positions) + one file at a time (~30 MB).
        _s = np.empty((total, 120, 8, 8), dtype=np.float16)
        _p = np.empty((total, 4672),       dtype=np.float16)
        _v = np.empty(total,               dtype=np.int64)
        _g = np.empty(total,               dtype=np.int32)   # per-position game id (= file)

        # Pass 2: fill pre-allocated arrays
        print(f"Loading {total:,} positions into RAM...")
        pos = 0
        skipped = 0
        thinned = 0
        game_id = 0
        for f, take in load_plan:
            try:
                with np.load(f, allow_pickle=True) as d:
                    s = d['states'][:take]
                    p = d['policies'][:take]
                    v = d['values'][:take]
                # Loud guard: a temperature=0 overflow once silently filled policy
                # targets with NaN, and the trainer's nan_to_num masked it. Fail fast
                # and visibly here instead of training on corrupt targets again.
                if not (np.isfinite(s).all() and np.isfinite(p).all()):
                    skipped += 1
                    continue
                # Decorrelate value targets at load time: cap positions/game so a long game's
                # many correlated positions don't over-weight one outcome (AlphaGo, Nature 2016).
                # The full .npz on disk is untouched.
                cap = MAX_POSITIONS_PER_GAME
                if (v == 1).all() and DRAW_MAX_POSITIONS > 0:
                    cap = DRAW_MAX_POSITIONS if cap == 0 else min(cap, DRAW_MAX_POSITIONS)
                if cap > 0 and len(v) > cap:
                    keep = np.unique(np.linspace(0, len(v) - 1, cap).astype(int))
                    thinned += len(v) - len(keep)
                    s, p, v = s[keep], p[keep], v[keep]
                    take = len(v)
                _s[pos:pos + take] = s
                _p[pos:pos + take] = p
                _v[pos:pos + take] = v.astype(np.int64)
                _g[pos:pos + take] = game_id   # all positions in one file = one game
                pos += take
                game_id += 1
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if skipped:
            print(f"  ⚠️ Skipped {skipped} file(s) with non-finite (NaN/Inf) data.")

        # Slice to actually-filled rows: a skipped or failed file must never leave
        # uninitialized np.empty garbage (which would train as NaN states / out-of-range
        # value labels) in the dataset.
        _s = _s[:pos]; _p = _p[:pos]; _v = _v[:pos]; _g = _g[:pos]
        self._states   = torch.from_numpy(_s)
        self._policies = torch.from_numpy(_p)
        self._values   = torch.from_numpy(_v)
        self._game_ids = _g            # numpy int32, one per position
        self.num_games = game_id
        self.total_positions = pos

        ram_gb = (self._states.nbytes + self._policies.nbytes + self._values.nbytes) / 1e9
        if DRAW_MAX_POSITIONS > 0 or MAX_POSITIONS_PER_GAME > 0:
            before = pos + thinned
            print(f"Subsample (draw_cap={DRAW_MAX_POSITIONS} game_cap={MAX_POSITIONS_PER_GAME}): "
                  f"{before:,} → {pos:,} positions "
                  f"(thinned {thinned:,}, {100*thinned/max(before,1):.0f}% of buffer)")
        print(f"Dataset Mapped: {self.total_positions:,} positions ({ram_gb:.1f} GB RAM).")

    def __len__(self):
        return self.total_positions

    def __getitem__(self, idx):
        state  = self._states[idx]
        policy = self._policies[idx]
        value  = self._values[idx]

        if torch.rand(1).item() < 0.5:
            # Horizontal (file a<->h) mirror is a valid chess symmetry EXCEPT it swaps
            # kingside<->queenside, so the castling-rights planes (112=WK,113=WQ,114=BK,
            # 115=BQ) must be swapped to match the mirrored geometry. Without this, every
            # position that still has castling rights becomes an inconsistent (illegal)
            # encoding. Piece/en-passant planes mirror correctly under flip; side-to-move
            # and clock planes are unaffected by a horizontal mirror.
            state  = torch.flip(state, dims=[2])
            state[[112, 113, 114, 115]] = state[[113, 112, 115, 114]]
            policy = policy[_H_FLIP_PERM]

        return {
            'state':  state.float(),
            'policy': policy.float(),
            'value':  value,
        }

def train_model(data_path="data/self_play",
                input_model_path="game_engine/model/best_model.pth",
                output_model_path="game_engine/model/candidate.pth",
                epochs=1,
                batch_size=256,
                lr=0.0001,
                window_size=20,
                total_iterations=1000):
    """
    Trains the model on data from data_path.
    Returns: (avg_policy_loss, avg_value_loss)
    """
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Plan the sliding window at FILE granularity (newest-iteration-first).
    plan = scan_window_files(data_path, window_size)           # [(path, n_positions)]
    total_window = sum(n for _, n in plan)
    if total_window == 0:
        print("Skipping training (No Data).")
        return 0.0, 0.0

    # Hold out ~10% of FILES as a disjoint val set. One .npz == one game, so a file-level split
    # IS a game-level split → no position from a val game leaks into train (honest val metrics).
    rng = np.random.default_rng(42)
    n_files = len(plan)
    n_val_files = max(1, int(n_files * 0.1)) if n_files > 1 else 0
    val_sel = set(rng.choice(n_files, size=n_val_files, replace=False).tolist()) if n_val_files else set()
    val_plan   = [plan[i] for i in range(n_files) if i in val_sel]
    train_plan = [plan[i] for i in range(n_files) if i not in val_sel]

    # RAM-aware chunking: partition the train files into chunks of <= TRAIN_CHUNK_POSITIONS each
    # (whole files only — a game is never split). If the whole window fits in one chunk this is a
    # single in-RAM load (old behaviour). If it's larger, every epoch sweeps ALL chunks
    # (load → train → free), so the entire last-`window_size` iterations get trained on instead
    # of the oldest positions being silently dropped by a hard cap.
    chunk_cap = int(os.environ.get("TRAIN_CHUNK_POSITIONS", 3_000_000))
    # Keep the persistent val set within one RAM chunk (10% of a very large window could itself
    # exceed RAM; trim by whole files if so).
    if sum(n for _, n in val_plan) > chunk_cap:
        trimmed, cum = [], 0
        for f, n in val_plan:
            if trimmed and cum + n > chunk_cap:
                break
            trimmed.append((f, n)); cum += n
        val_plan = trimmed
    train_chunks, cur, cur_n = [], [], 0
    for f, n in train_plan:
        if cur and cur_n + n > chunk_cap:
            train_chunks.append(cur); cur, cur_n = [], 0
        cur.append((f, n)); cur_n += n
    if cur:
        train_chunks.append(cur)
    if not train_chunks:
        print("Skipping training (no train files after val split).")
        return 0.0, 0.0
    multi_chunk = len(train_chunks) > 1
    n_train_pos = sum(n for _, n in train_plan)
    n_val_pos   = sum(n for _, n in val_plan)
    print(f"Window: {total_window:,} pos / {n_files} files → {n_train_pos:,} train / {n_val_pos:,} val"
          f"  |  {len(train_chunks)} chunk(s) ≤{chunk_cap:,} pos" + ("  [CHUNKED]" if multi_chunk else ""))

    # DataLoader: during training the self-play server is dead so all vCPUs are free. Workers
    # only do flip-aug + f16→f32 cast over the read-only f16 buffer (COW fork → no RAM ×).
    # TRAIN_DL_WORKERS (default cores/2, ≤16 → 16 on the L4 box) · TRAIN_DL_PREFETCH (default 4).
    num_dl_workers = int(os.environ.get("TRAIN_DL_WORKERS", min(16, max(2, (os.cpu_count() or 4) // 2))))
    dl_prefetch = int(os.environ.get("TRAIN_DL_PREFETCH", 4))

    def _mk_loader(ds, shuffle, persistent):
        kw = dict(num_workers=num_dl_workers, pin_memory=(num_dl_workers > 0),
                  persistent_workers=(persistent and num_dl_workers > 0))
        if num_dl_workers > 0:
            kw['prefetch_factor'] = dl_prefetch
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)

    # Val set loaded once, persists across all epochs. Single-chunk: load the train data once and
    # reuse it every epoch (no reload). Multi-chunk: train loaders are built per chunk in the loop.
    val_dataset = ChessDataset(data_path, file_plan=val_plan)
    val_dataloader = _mk_loader(val_dataset, shuffle=False, persistent=True) if len(val_dataset) > 0 else None
    train_dataset = train_dataloader = None
    if not multi_chunk:
        train_dataset = ChessDataset(data_path, file_plan=train_chunks[0])
        train_dataloader = _mk_loader(train_dataset, shuffle=True, persistent=True)

    # 2. Load Model
    model = ChessCNN().to(device)
    checkpoint = None
    anchor = None
    KL_ANCHOR_BETA = float(os.environ.get("KL_ANCHOR_BETA", 0.0))

    if os.path.exists(input_model_path):
        print(f"Loading training base from {input_model_path}")
        try:
            raw = torch.load(input_model_path, map_location=device)
            if 'model_state_dict' in raw:
                sd = raw['model_state_dict']
                checkpoint = raw
            elif 'state_dict' in raw:
                sd = raw['state_dict']
                checkpoint = raw
            else:
                sd = raw  # legacy bare state dict — no optimizer/scheduler to restore
            # Strip torch.compile prefix (_orig_mod.) if present
            if any(k.startswith('_orig_mod.') for k in sd):
                sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
            model.load_state_dict(sd)
            # KL-anchor: a frozen copy of the pretrained base. Its policy is the prior the
            # candidate is penalized for drifting from (anti-forgetting on small early data).
            if KL_ANCHOR_BETA > 0:
                anchor = ChessCNN().to(device)
                anchor.load_state_dict(sd)
                anchor.eval()
                for p in anchor.parameters():
                    p.requires_grad_(False)
                print(f"KL-anchor ON: beta={KL_ANCHOR_BETA} (frozen pretrained prior)")
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    else:
        print(f"No existing model at {input_model_path}, starting from random weights.")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # scheduler.step() is called once per batch. A training phase runs epochs × (all chunks),
    # so batches_per_epoch sums every chunk's batch count; T_max must include the epochs factor
    # or the cosine completes its half-period ~epochs× too early (CosineAnnealingLR is periodic).
    batches_per_epoch = sum(max(1, (sum(n for _, n in ch) + batch_size - 1) // batch_size)
                            for ch in train_chunks)
    total_steps = total_iterations * epochs * max(batches_per_epoch, 1)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    if checkpoint is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Scheduler restored: step {scheduler.last_epoch}/{scheduler.T_max} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # 3. Loss Functions
    ce_value_loss = nn.CrossEntropyLoss()
    
    # --- AMP Scaler ---
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Header ─────────────────────────────────────────────────────────────────
    n_val_batches = len(val_dataloader) if val_dataloader is not None else 0
    sep = '─' * 86
    print(f"\n{'═'*86}")
    print(f"  RL Training  |  {epochs} epochs  |  batch={batch_size}"
          f"  |  {n_train_pos} train / {n_val_pos} val positions"
          f"  |  value_weight={VALUE_LOSS_WEIGHT}")
    print(f"  {batches_per_epoch} train batches/ep  |  {n_val_batches} val batches/ep"
          f"  |  {num_dl_workers} workers ×{dl_prefetch} prefetch  |  {len(train_chunks)} chunk(s)")
    print(f"{'═'*86}")
    print(f"{'Ep':>6}  {'Tr-P':>7}  {'Tr-V':>7}  {'Va-P':>7}  {'Va-V':>7}"
          f"  {'Tr-Acc':>7}  {'Va-Acc':>7}  {'GNorm':>7}  {'LR':>10}  {'Time':>5}")
    print(sep)
    sys.stdout.flush()

    last_p_loss = 0.0
    last_v_loss = 0.0
    best_val    = float('inf')

    def _train_batches():
        """Yield every train batch for one epoch. Single-chunk: the one persistent loader.
        Multi-chunk: load → yield → free each chunk (shuffled order) so only one chunk is in RAM
        at a time, yet the whole `window_size`-iteration window is still covered every epoch."""
        if multi_chunk:
            order = list(range(len(train_chunks)))
            np.random.shuffle(order)
            for ci in order:
                ds = ChessDataset(data_path, file_plan=train_chunks[ci])
                if len(ds) == 0:
                    continue
                dl = _mk_loader(ds, shuffle=True, persistent=False)
                for b in dl:
                    yield b
                del dl, ds
                gc.collect()
        else:
            for b in train_dataloader:
                yield b

    # 4. Training Loop
    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train pass ─────────────────────────────────────────────────────────
        model.train()
        tr_p = tr_v = tr_acc = g_norm = tr_kl = 0.0
        n_tr = 0

        train_bar = tqdm(_train_batches(), total=batches_per_epoch,
                         desc=f"  train {epoch+1}/{epochs}", leave=False, ncols=88, unit='bat')
        for batch in train_bar:
            states          = batch['state'].to(device, non_blocking=True)
            target_policies = batch['policy'].to(device, non_blocking=True)
            target_values   = batch['value'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Anchor logits FIRST (no_grad/fp16), freed before training activations accumulate,
            # so only the (B,4672) log-probs (~38 MB) persist — no VRAM overlap with backward.
            anchor_logp = None
            if anchor is not None:
                with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                    a_logits, _ = anchor(states)
                anchor_logp = torch.log_softmax(a_logits.float(), dim=1)
                del a_logits

            with torch.amp.autocast(device_type=device.type):
                pred_policies, pred_values = model(states)
                v_loss = ce_value_loss(pred_values, target_values)

            # Policy loss in fp32: pred_policies may be fp16 inside autocast and
            # fp16 softmax can overflow to NaN (not just -inf). clamp handles -inf→-100
            # and nan_to_num handles any remaining NaN in log_probs or sparse targets.
            log_probs = torch.log_softmax(pred_policies.float(), dim=1).clamp(min=-100.0)
            p_loss = -(target_policies.nan_to_num(0.0) * log_probs).sum(dim=1).mean()
            loss   = p_loss + VALUE_LOSS_WEIGHT * v_loss

            if anchor_logp is not None:
                # KL(anchor ‖ candidate): hold the policy near the pretrained prior.
                kl = (anchor_logp.exp() * (anchor_logp - log_probs)).sum(dim=1).mean()
                loss = loss + KL_ANCHOR_BETA * kl
                tr_kl += float(kl.item())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            acc = (pred_policies.detach().argmax(1) == target_policies.argmax(1)).float().mean().item()
            tr_p   += p_loss.item()
            tr_v   += v_loss.item()
            tr_acc += acc
            # On an AMP fp16 overflow the GradScaler skips the step (training is protected),
            # but clip_grad_norm returns inf/nan for that batch — don't let it poison the
            # epoch's averaged GNorm display.
            tn = total_norm.item()
            g_norm += tn if np.isfinite(tn) else 0.0
            n_tr   += 1
            train_bar.set_postfix(p=f"{p_loss.item():.3f}", v=f"{v_loss.item():.3f}",
                                   acc=f"{acc*100:.1f}%")

        # ── Val pass ───────────────────────────────────────────────────────────
        model.eval()
        va_p = va_v = va_acc = 0.0
        n_va = 0

        val_bar = tqdm(val_dataloader if val_dataloader is not None else [],
                       desc=f"  val   {epoch+1}/{epochs}", leave=False, ncols=88, unit='bat')
        with torch.no_grad():
            for batch in val_bar:
                states          = batch['state'].to(device, non_blocking=True)
                target_policies = batch['policy'].to(device, non_blocking=True)
                target_values   = batch['value'].to(device, non_blocking=True)

                # Validation in fp32 (NO autocast): val has no GradScaler, so an fp16 logit
                # overflow during early-training instability would surface as a NaN val loss
                # (seen at epoch 2). fp32 val is cheap (no backward) and overflow-proof.
                pred_policies, pred_values = model(states)
                v_loss = ce_value_loss(pred_values, target_values)

                log_probs = torch.log_softmax(pred_policies.float(), dim=1).clamp(min=-100.0)
                p_loss = -(target_policies.nan_to_num(0.0) * log_probs).sum(dim=1).mean()

                acc = (pred_policies.argmax(1) == target_policies.argmax(1)).float().mean().item()
                va_p   += p_loss.item()
                va_v   += v_loss.item()
                va_acc += acc
                n_va   += 1

        # ── Epoch row ──────────────────────────────────────────────────────────
        tr_p_avg   = tr_p  / max(n_tr, 1)
        tr_v_avg   = tr_v  / max(n_tr, 1)
        va_p_avg   = va_p  / max(n_va, 1)
        va_v_avg   = va_v  / max(n_va, 1)
        tr_acc_pct = tr_acc / max(n_tr, 1) * 100
        va_acc_pct = va_acc / max(n_va, 1) * 100
        gn_avg     = g_norm / max(n_tr, 1)
        kl_avg     = tr_kl  / max(n_tr, 1)
        lr         = scheduler.get_last_lr()[0]
        elapsed    = int(time.time() - epoch_start)

        val_total = va_p_avg + va_v_avg
        marker = ' ↑' if val_total < best_val else ''
        if val_total < best_val:
            best_val = val_total

        print(f"{epoch+1:>3}/{epochs:<3}  {tr_p_avg:>7.4f}  {tr_v_avg:>7.4f}"
              f"  {va_p_avg:>7.4f}  {va_v_avg:>7.4f}"
              f"  {tr_acc_pct:>6.1f}%  {va_acc_pct:>6.1f}%"
              f"  {gn_avg:>7.3f}  {lr:>10.2e}  {elapsed:>4}s{marker}"
              f"{'  KL='+format(kl_avg,'.3f') if anchor is not None else ''}")
        sys.stdout.flush()

        last_p_loss = tr_p_avg
        last_v_loss = tr_v_avg

    # 5. Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch + 1,
        'val_policy_loss': va_p_avg,
        'val_value_loss': va_v_avg,
    }, output_model_path)
    print(f"New model saved to {output_model_path}")

    # Free the in-RAM dataset (up to ~100 GB at the 20-iter window) and the model/optimizer
    # GPU memory, and shut down the persistent DataLoader workers, BEFORE returning to the
    # eval phase (which spawns its own inference servers). Otherwise the dataset RAM + model
    # VRAM linger via the persistent workers until a later GC.
    if train_dataloader is not None:
        del train_dataloader, train_dataset
    if val_dataloader is not None:
        del val_dataloader, val_dataset
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return last_p_loss, last_v_loss

if __name__ == "__main__":
    train_model(epochs=10)