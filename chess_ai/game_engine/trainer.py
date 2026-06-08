import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import glob
import sys
import time
import bisect
import collections

# Ensure we can import from the parent directory
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN

# Named tuple to store the file boundaries
FileIndex = collections.namedtuple('FileIndex', ['file_path', 'start_idx', 'end_idx'])

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

class ChessDataset(Dataset):
    def __init__(self, data_dir, window_size=20):
        self.file_map = []  # Stores (file_path, start_index, end_index)
        self.total_positions = 0
        
        if not os.path.exists(data_dir):
            print(f"Warning: No data found in {data_dir}")
            return

        # --- FOLDER-BASED SLIDING WINDOW LOGIC ---
        # 1. Identify all 'iter_X' folders
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("iter_")]
        
        # 2. Sort by iteration number (ascending)
        try:
            sorted_subdirs = sorted(subdirs, key=lambda x: int(x.split("_")[1]))
        except ValueError:
            print("Warning: Could not parse iteration folders.")
            sorted_subdirs = []

        # 3. Select the last N folders (Sliding Window)
        active_folders = sorted_subdirs[-window_size:] if sorted_subdirs else []
        
        if not active_folders:
            print("No iteration data folders found.")
            return

        print(f"Data Window: Training on last {len(active_folders)} iterations: {active_folders[0]} to {active_folders[-1]}")

        # 4. Collect all .npz files inside these folders
        all_files = []
        for folder in active_folders:
            full_path = os.path.join(data_dir, folder)
            all_files.extend(glob.glob(os.path.join(full_path, "*.npz")))

        print(f"Mapping {len(all_files)} data files...")
        
        for f in all_files:
            try:
                # Use np.load with mmap_mode to quickly check shape without loading data
                with np.load(f, allow_pickle=True) as data:
                    s_shape = data['states'].shape
                    p_shape = data['policies'].shape
                    
                    num_positions = s_shape[0]

                    # Validation: Check shapes
                    if len(s_shape) != 4 or s_shape[1:] != (120, 8, 8):
                        print(f"Skipping corrupt file {f}: State shape {s_shape} != (120, 8, 8)")
                        continue
                        
                    if len(p_shape) != 2 or p_shape[1] != 4672:
                        print(f"Skipping corrupt file {f}: Policy shape {p_shape}")
                        continue
                    
                    if num_positions > 0:
                        start_idx = self.total_positions
                        end_idx = self.total_positions + num_positions - 1
                        
                        self.file_map.append(FileIndex(f, start_idx, end_idx))
                        self.total_positions += num_positions
                        
            except Exception as e:
                print(f"Error checking {f}: {e}")
                continue

        # Precompute start indices for O(log N) bisect lookup
        self._start_idxs = [fi.start_idx for fi in self.file_map]

        # Cache for the currently loaded file
        self._current_file = None
        self._current_data = None
        
        print(f"Dataset Mapped: {self.total_positions} positions.")

    def __len__(self):
        return self.total_positions

    def _load_file_for_index(self, idx):
        # O(log N) bisect lookup
        i = bisect.bisect_right(self._start_idxs, idx) - 1
        if i < 0 or i >= len(self.file_map):
            raise IndexError(f"Index {idx} out of range for file map.")
        file_info = self.file_map[i]
        target_file = file_info.file_path
        local_idx = idx - file_info.start_idx

        # Check if the file is already cached
        if target_file != self._current_file:
            with np.load(target_file, allow_pickle=True) as data:
                self._current_data = {
                    'states':   torch.from_numpy(data['states'].copy()),
                    'policies': torch.from_numpy(data['policies'].copy()),
                    'values':   torch.from_numpy(data['values'].astype(np.int64)),
                }
            self._current_file = target_file

        return self._current_data, local_idx


    def __getitem__(self, idx):
        data, local_idx = self._load_file_for_index(idx)

        state = data['states'][local_idx]
        policy = data['policies'][local_idx]
        value = data['values'][local_idx]

        if torch.rand(1).item() < 0.5:
            state = torch.flip(state, dims=[2])   # mirror a-file ↔ h-file
            policy = policy[_H_FLIP_PERM]

        return {
            'state': state.float(),
            'policy': policy.float(),
            'value': value,  # int64 class index: 0=win, 1=draw, 2=loss
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

    # 1. Prepare Data with Sliding Window
    dataset = ChessDataset(data_path, window_size=window_size)
    if len(dataset) == 0:
        print("Skipping training (No Data).")
        return 0.0, 0.0

    num_dl_workers = min(4, max(1, (os.cpu_count() or 4) // 4))

    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_dl_workers, pin_memory=True,
        persistent_workers=(num_dl_workers > 0),
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_dl_workers, pin_memory=True,
        persistent_workers=(num_dl_workers > 0),
    )

    # 2. Load Model
    model = ChessCNN().to(device)
    checkpoint = None

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
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    else:
        print(f"No existing model at {input_model_path}, starting from random weights.")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = total_iterations * len(dataloader)
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
    n_train = len(dataloader)
    n_val   = len(val_dataloader)
    sep = '─' * 86
    print(f"\n{'═'*86}")
    print(f"  RL Training  |  {epochs} epochs  |  batch={batch_size}"
          f"  |  {train_size} train / {val_size} val positions")
    print(f"  {n_train} train batches/ep  |  {n_val} val batches/ep"
          f"  |  {num_dl_workers} DataLoader workers")
    print(f"{'═'*86}")
    print(f"{'Ep':>6}  {'Tr-P':>7}  {'Tr-V':>7}  {'Va-P':>7}  {'Va-V':>7}"
          f"  {'Tr-Acc':>7}  {'Va-Acc':>7}  {'GNorm':>7}  {'LR':>10}  {'Time':>5}")
    print(sep)
    sys.stdout.flush()

    last_p_loss = 0.0
    last_v_loss = 0.0
    best_val    = float('inf')

    # 4. Training Loop
    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train pass ─────────────────────────────────────────────────────────
        model.train()
        tr_p = tr_v = tr_acc = g_norm = 0.0
        n_tr = 0

        train_bar = tqdm(dataloader, desc=f"  train {epoch+1}/{epochs}",
                         leave=False, ncols=88, unit='bat')
        for batch in train_bar:
            states          = batch['state'].to(device, non_blocking=True)
            target_policies = batch['policy'].to(device, non_blocking=True)
            target_values   = batch['value'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type):
                pred_policies, pred_values = model(states)
                v_loss = ce_value_loss(pred_values, target_values)

            # Policy loss in fp32: pred_policies may be fp16 inside autocast and
            # fp16 softmax can overflow to NaN (not just -inf). clamp handles -inf→-100
            # and nan_to_num handles any remaining NaN in log_probs or sparse targets.
            log_probs = torch.log_softmax(pred_policies.float(), dim=1).clamp(min=-100.0)
            p_loss = -(target_policies.nan_to_num(0.0) * log_probs).sum(dim=1).mean()
            loss   = v_loss + p_loss

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
            g_norm += total_norm.item()
            n_tr   += 1
            train_bar.set_postfix(p=f"{p_loss.item():.3f}", v=f"{v_loss.item():.3f}",
                                   acc=f"{acc*100:.1f}%")

        # ── Val pass ───────────────────────────────────────────────────────────
        model.eval()
        va_p = va_v = va_acc = 0.0
        n_va = 0

        val_bar = tqdm(val_dataloader, desc=f"  val   {epoch+1}/{epochs}",
                       leave=False, ncols=88, unit='bat')
        with torch.no_grad():
            for batch in val_bar:
                states          = batch['state'].to(device, non_blocking=True)
                target_policies = batch['policy'].to(device, non_blocking=True)
                target_values   = batch['value'].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
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
        lr         = scheduler.get_last_lr()[0]
        elapsed    = int(time.time() - epoch_start)

        val_total = va_p_avg + va_v_avg
        marker = ' ↑' if val_total < best_val else ''
        if val_total < best_val:
            best_val = val_total

        print(f"{epoch+1:>3}/{epochs:<3}  {tr_p_avg:>7.4f}  {tr_v_avg:>7.4f}"
              f"  {va_p_avg:>7.4f}  {va_v_avg:>7.4f}"
              f"  {tr_acc_pct:>6.1f}%  {va_acc_pct:>6.1f}%"
              f"  {gn_avg:>7.3f}  {lr:>10.2e}  {elapsed:>4}s{marker}")
        sys.stdout.flush()

        last_p_loss = tr_p_avg
        last_v_loss = tr_v_avg

    # 5. Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, output_model_path)
    print(f"New model saved to {output_model_path}")
    
    return last_p_loss, last_v_loss

if __name__ == "__main__":
    train_model(epochs=10)