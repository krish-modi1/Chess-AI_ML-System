import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Import AMP for Mixed Precision Training
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import os
import glob
import sys
import time
import collections
from tqdm import tqdm

# Ensure we can import from the parent directory
sys.path.append(os.getcwd())

from game_engine.cnn import ChessCNN

# Named tuple to store the file boundaries
FileIndex = collections.namedtuple('FileIndex', ['file_path', 'start_idx', 'end_idx'])

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
                    if len(s_shape) != 4 or s_shape[1:] != (16, 8, 8):
                        print(f"Skipping corrupt file {f}: State shape {s_shape} != (16, 8, 8)")
                        continue
                        
                    if len(p_shape) != 2 or p_shape[1] != 8192:
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

        # Cache for the currently loaded file
        self._current_file = None
        self._current_data = None
        
        print(f"Dataset Mapped: {self.total_positions} positions.")

    def __len__(self):
        return self.total_positions

    def _load_file_for_index(self, idx):
        # Determine which file the global index 'idx' belongs to
        
        # Binary search or simple iteration (simple iteration is fine for typical file counts)
        target_file = None
        local_idx = 0
        
        for file_info in self.file_map:
            if file_info.start_idx <= idx <= file_info.end_idx:
                target_file = file_info.file_path
                local_idx = idx - file_info.start_idx
                break

        if target_file is None:
            raise IndexError(f"Index {idx} out of range for file map.")
            
        # Check if the file is already cached
        if target_file != self._current_file:
            # print(f"Loading new data file: {target_file}")
            data = np.load(target_file, allow_pickle=True)
            self._current_data = {
                'states': torch.from_numpy(data['states']),
                'policies': torch.from_numpy(data['policies']),
                'values': torch.from_numpy(data['values']),
            }
            self._current_file = target_file
            
        return self._current_data, local_idx


    def __getitem__(self, idx):
        # FIX E: Load data lazily based on index
        data, local_idx = self._load_file_for_index(idx)
        
        return {
            'state': data['states'][local_idx].float(),
            'policy': data['policies'][local_idx].float(),
            'value': data['values'][local_idx].float()
        }

def train_model(data_path="data/self_play", 
                input_model_path="game_engine/model/best_model.pth", 
                output_model_path="game_engine/model/candidate.pth", 
                epochs=1, 
                batch_size=256, 
                lr=0.0001,
                window_size=20):
    """
    Trains the model on data from data_path.
    Returns: (avg_policy_loss, avg_value_loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Prepare Data with Sliding Window
    dataset = ChessDataset(data_path, window_size=window_size)
    if len(dataset) == 0:
        print("Skipping training (No Data).")
        return 0.0, 0.0

    # Optimization: num_workers > 0 and pin_memory=True for faster GPU transfer
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        prefetch_factor=4,          
        pin_memory=True,            
        persistent_workers=True  
    )

    # 2. Load Model
    model = ChessCNN(upgraded=True).to(device)
    
    # Load existing weights (The Champion) to continue training
    if os.path.exists(input_model_path):
        print(f"Loading training base from {input_model_path}")
        try:
            checkpoint = torch.load(input_model_path, map_location=device)
            # Robust Dictionary Loading
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    else:
        print(f"No existing model at {input_model_path}, starting from random weights.")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # Add exponential LR decay for controlled convergence
    scheduler = ExponentialLR(optimizer, gamma=0.7)  # Halves LR each epoch
    
    # 3. Loss Functions
    mse_loss = nn.MSELoss()
    
    # --- AMP Scaler ---
    scaler = GradScaler()

    last_p_loss = 0.0
    last_v_loss = 0.0

    # 4. Training Loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        p_loss_total = 0
        v_loss_total = 0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            states = batch['state'].to(device, non_blocking=True)
            target_policies = batch['policy'].to(device, non_blocking=True)
            target_values = batch['value'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # --- Mixed Precision Forward Pass ---
            with autocast():
                # cnn returns (policy_logits, value)
                pred_policies, pred_values = model(states)
                
                # --- Value Loss (MSE) ---
                v_loss = mse_loss(pred_values.squeeze(), target_values)
                
                # --- Policy Loss (Cross Entropy) ---
                log_probs = torch.log_softmax(pred_policies, dim=1)
                p_loss = -(target_policies * log_probs).sum(dim=1).mean()
                
                # Total Loss
                loss = v_loss + p_loss
            
            # --- Scaled Backward Pass ---
            scaler.scale(loss).backward()

            # Unscale and clip gradients for stability
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            p_loss_total += p_loss.item()
            v_loss_total += v_loss.item()
            batch_count += 1

            # SIMPLE LOGGING FIX
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / batch_count
                elapsed = time.time() - epoch_start_time
                eta_secs = int((elapsed / (batch_idx + 1)) * (len(dataloader) - batch_idx - 1))
                eta_str = f"{eta_secs//60}m {eta_secs%60}s" if eta_secs >= 60 else f"{eta_secs}s"
                print(f"Batch {batch_idx+1}/{len(dataloader)} Loss {avg_loss:.4f} ETA {eta_str}")

            # Optional: Log gradient norm occasionally (every 50 batches)
            if (batch_idx + 1) % 50 == 0 and batch_idx > 0:
                print(f"  Grad Norm: {total_norm:.4f}")

        
        if batch_count > 0:
            last_p_loss = p_loss_total / batch_count
            last_v_loss = v_loss_total / batch_count

            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/batch_count:.4f} (Pol: {last_p_loss:.4f} Val: {last_v_loss:.4f})")

            # Step LR scheduler (aggressive decay)
            if epoch < 3:  # Only decay for first 2 epochs
                scheduler.step()
                print(f"LR decayed to: {scheduler.get_last_lr()[0]:.6f}")

    # 5. Save Model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_model_path)
    print(f"New model saved to {output_model_path}")
    
    return last_p_loss, last_v_loss

if __name__ == "__main__":
    train_model(epochs=10)