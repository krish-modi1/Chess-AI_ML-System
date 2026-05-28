import json
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure imports work if needed, though this script is mostly standalone
sys.path.append(os.getcwd())

METRICS_FILE = "game_engine/model/metrics.json"
OUTPUT_DIR = "game_engine/model"

def plot_metrics():
    if not os.path.exists(METRICS_FILE):
        print(f"No metrics file found at {METRICS_FILE}")
        return
    
    history = []
    
    # Retry logic to handle potential race conditions if main.py is writing
    for _ in range(3):
        try:
            with open(METRICS_FILE, 'r') as f:
                history = json.load(f)
            break
        except json.JSONDecodeError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error reading metrics file: {e}")
            return
    
    if not history:
        print("Metrics file is empty.")
        return
    
    iterations = [h['iteration'] for h in history]
    p_loss = [h.get('policy_loss', 0.0) for h in history]
    v_loss = [h.get('value_loss', 0.0) for h in history]
    win_rates = [h.get('arena_win_rate', 0.0) for h in history]
    
    # Filter for ELOs (might be None in some entries)
    elo_data = [(h['iteration'], h['elo']) for h in history if h.get('elo') is not None]
    elo_iters = [d[0] for d in elo_data]
    elos = [d[1] for d in elo_data]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create 4 separate figures with high resolution and transparent backgrounds
    dpi = 150
    facecolor = 'none'  # Transparent background
    edgecolor = 'none'
    
    # 1. Policy Loss Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    ax.set_facecolor('none')  # Transparent axes background
    ax.plot(iterations, p_loss, label='Policy Loss', color='#3b82f6', linewidth=2.5, marker='o', markersize=4)
    ax.set_title('Policy Loss vs Iterations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Policy Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    policy_output = os.path.join(OUTPUT_DIR, 'policy_loss.png')
    fig.savefig(policy_output, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f"Policy loss plot saved to {policy_output}")
    plt.close(fig)
    
    # 2. Validation Loss Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    ax.set_facecolor('none')  # Transparent axes background
    ax.plot(iterations, v_loss, label='Validation Loss', color='#10b981', linewidth=2.5, marker='o', markersize=4)
    ax.set_title('Validation Loss vs Iterations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    value_output = os.path.join(OUTPUT_DIR, 'validation_loss.png')
    fig.savefig(value_output, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f"Validation loss plot saved to {value_output}")
    plt.close(fig)
    
    # 3. Candidate Win Rate Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    ax.set_facecolor('none')  # Transparent axes background
    ax.plot(iterations, win_rates, label='Win Rate vs Prev Best', color='#f59e0b', linewidth=2.5, marker='o', markersize=4)
    ax.axhline(y=0.55, color='#ef4444', linestyle='--', linewidth=2, label='Promotion Threshold (0.55)')
    ax.set_title('Candidate Win Rate vs Iterations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    winrate_output = os.path.join(OUTPUT_DIR, 'candidate_winrate.png')
    fig.savefig(winrate_output, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f"Candidate win rate plot saved to {winrate_output}")
    plt.close(fig)
    
    # 4. Elo Rating Plot
    if elos:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
        ax.set_facecolor('none')  # Transparent axes background
        ax.plot(elo_iters, elos, label='Est. Elo (vs Stockfish)', color='#8b5cf6', linewidth=2.5, marker='o', markersize=2, markeredgewidth=2)
        ax.set_title('Estimated Elo Rating Progression', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Elo Rating', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        elo_output = os.path.join(OUTPUT_DIR, 'elo_rating.png')
        fig.savefig(elo_output, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Elo rating plot saved to {elo_output}")
        plt.close(fig)
    else:
        print("No Elo data available - skipping Elo plot")
    
    print("\nAll 4 plots generated successfully with transparent backgrounds!")

if __name__ == "__main__":
    plot_metrics()
