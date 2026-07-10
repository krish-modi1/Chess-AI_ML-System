"""plot_metrics.py — training telemetry plotting suite.

Reads game_engine/model/metrics.json and renders a paper-grade view of the
self-play RL run, following the plotting conventions used in the AlphaZero /
KataGo / RLHF literature:

  * Elo progression is the headline (with CI error bars where measured, promotion
    stars, carried-forward champion Elo shown distinctly, and the Stockfish anchor
    annotated) — the only metric that is comparable across iterations.
  * Train-vs-val curves are overlaid so the generalization gap is visible.
  * KL-to-reference (the β=1.0 anchor) gets its own panel — in RLHF this is THE
    constraint plot; here it shows the policy staying pinned to the pretrained prior.
  * The MCTS-vs-net search metrics (top1-agree / override / KL / Δentropy) form the
    AlphaZero "policy improvement from search" panel.

The arena is no longer a gate (the Stockfish-anchored promotion gate replaced it — see
main.py STOCKFISH_GATE), so arena win-rate / reward-KL panels were removed: Elo vs Stockfish
is the sole progress metric, alongside the training and offline-probe panels.

Outputs a master dashboard.png plus standalone high-DPI transparent panels.
Run from chess_ai/ (paths resolve relative to this file regardless of cwd):
    python plot_metrics.py
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# Paths anchored to this file so it works from repo root or chess_ai/.
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(HERE, "game_engine", "model", "metrics.json")
OUTPUT_DIR = os.path.join(HERE, "game_engine", "evaluation", "plots")
# Promotion gate is ERA-AWARE: metrics.json has no per-iter "promoted" field, so the plot infers a
# promotion from win_rate >= gate. The gate changed 0.55 → 0.50 at iter 89 (Session 12 — unfreeze the
# frozen self-play generator; KataGo's 100/200 gate). Using a single constant would retroactively mark
# old 0.50-0.55 rejections as promotions (dishonest). GATE_HISTORY = (first_iter, gate), ascending.
# Adjust the 89 boundary if the box picked up the new config at a different iteration.
GATE_HISTORY = [(0, 0.55), (89, 0.50)]


def gate_for(iteration):
    g = GATE_HISTORY[0][1]
    for start, val in GATE_HISTORY:
        if iteration >= start:
            g = val
    return g


DPI = 150
# Supervised-pretrained base = iteration 0. Trained on Lichess human games (~1800),
# measured ~1500 Elo vs Stockfish-1800 before any self-play. Loss values are read
# from the pretrained.pth checkpoint metadata (epoch 7) and are on the SAME WDL-CE
# scale as the RL run, so they plot directly. Policy top-1 acc ~57% (SL run, notes).
# KL-anchor = 0: the pretrained net IS the reference policy, so KL-to-reference is 0
# by definition. Prepended to history in main().
PRETRAIN_RECORD = {
    "iteration": 0,
    "model_elo": 1500,
    "elo_measured": True,
    "stockfish_elo": 1800,
    "val_policy_loss": 1.3657,   # pretrained.pth (epoch 7)
    "val_value_loss": 0.7603,    # pretrained.pth (epoch 7)
    "val_acc": 57,
    "kl_anchor": 0.0,
}

# Consistent, colorblind-aware palette reused across the dashboard and panels.
C = {
    "policy": "#3b82f6",   # blue
    "value": "#10b981",    # green
    "val": "#94a3b8",      # slate (validation / secondary)
    "elo": "#8b5cf6",      # violet
    "winrate": "#f59e0b",  # amber
    "reject": "#ef4444",   # red
    "kl": "#ec4899",       # pink
    "search": "#06b6d4",   # cyan
    "grad": "#64748b",     # slate-600
    "champ": "#f97316",    # orange
    "cand": "#6366f1",     # indigo
    "marker": "#6b7280",   # gray — intervention markers
}

# ----------------------------------------------------------------------------
# Intervention registry — manual levers we pulled, marked on the plots so a
# trajectory change is never mistaken for organic improvement (honesty first).
#   iter   = the iteration at which the change first TOOK EFFECT
#   label  = short caption drawn at the vertical line
#   panels = "all" (every iteration-x-axis panel) or a set of panel keys
#            ("elo","loss","acc","winrate","kl","search","value","grad")
# Add a line here whenever a lever is pulled.
# ----------------------------------------------------------------------------
INTERVENTIONS = [
    {"iter": 17, "label": "LR 1e-4→3e-4", "panels": "all"},
    # Reference reset: KL anchor pretrained-1800 → live champion. Set "iter" to the FIRST iteration
    # whose training log shows "KL anchor = CHAMPION" (adjust if it ships at a different iter than 22).
    {"iter": 22, "label": "anchor→champion", "panels": {"kl", "elo", "winrate", "loss"}},
    # Value-head architecture fix: 1-ch bottleneck → KataGo global avg+max pool WDL head (migrated via
    # grow_value_head.py). Negative result — value stayed flat. Mark the value/loss panels.
    {"iter": 85, "label": "value-head GP", "panels": {"value", "loss", "elo"}},
    # TD (z+Q) value target λ=0.3 + moves-left aux head go live. Phases in over ~window_size iters.
    {"iter": 87, "label": "TD λ=0.3 + aux", "panels": {"value", "loss"}},
    # Promotion gate 0.55 → 0.50 (unfreeze the self-play generator). Keep in sync with GATE_HISTORY.
    {"iter": 89, "label": "gate 0.55→0.50", "panels": {"winrate", "elo"}},
    # Lineage OFF (stop the monotonic drift) + train window 20→40. Fresh retrain from champion each iter.
    {"iter": 91, "label": "lineage off + win40", "panels": {"winrate", "elo", "loss", "value"}},
    # Stockfish-anchored gate replaces the arena (arena WR now diagnostic-only); training knobs reverted.
    # Champion re-promoted from the it56 freeze. Arena↔promotion decouple here (green dots may sit low).
    {"iter": 92, "label": "SF gate + promote", "panels": {"winrate", "elo"}},
]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.family": "sans-serif",
    })


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        print(f"No metrics file found at {METRICS_FILE}")
        return []
    # Retry — main.py may be mid-write.
    for _ in range(3):
        try:
            with open(METRICS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error reading metrics file: {e}")
            return []
    return []


def col(history, key, zero_is_missing=False):
    """Extract a numeric column as a float array; None (and optionally 0.0) → NaN
    so matplotlib draws a gap instead of a spurious spike to zero."""
    out = []
    for h in history:
        v = h.get(key)
        if v is None or (zero_is_missing and v == 0.0):
            out.append(np.nan)
        else:
            out.append(float(v))
    return np.array(out)


def parse(history):
    it = np.array([h["iteration"] for h in history])

    def _promoted(h):
        # Under the Stockfish-anchored gate (iter-92+), promotion = cand >= champ vs Stockfish, NOT the
        # arena WR. Prefer the sf_gate_* markers when present; fall back to the legacy arena-WR gate.
        cw, hw = h.get("sf_gate_cand_wr"), h.get("sf_gate_champ_wr")
        if cw is not None and hw is not None:
            return cw >= hw
        return h.get("arena_win_rate", 0.0) >= gate_for(h["iteration"])
    promoted = np.array([_promoted(h) for h in history])
    return {
        "it": it,
        "promoted": promoted,
        "p_loss": col(history, "policy_loss", zero_is_missing=True),
        "v_loss": col(history, "value_loss", zero_is_missing=True),
        "vp_loss": col(history, "val_policy_loss"),
        "vv_loss": col(history, "val_value_loss"),
        "train_acc": col(history, "train_acc"),
        "val_acc": col(history, "val_acc"),
        "kl_anchor": col(history, "kl_anchor"),
        "grad_norm": col(history, "grad_norm"),
        "elo": col(history, "model_elo"),
        "elo_ci": col(history, "elo_ci"),
        "elo_measured": np.array([bool(h.get("elo_measured")) for h in history]),
        "sf_elo": col(history, "stockfish_elo"),
        "va_cand": col(history, "value_acc_cand"),
        "va_champ": col(history, "value_acc_champ"),
        "dec_cand": col(history, "value_decisive_cand"),
        "dec_champ": col(history, "value_decisive_champ"),
        "s_top1": col(history, "search_top1"),
        "s_override": col(history, "search_override"),
        "s_kl": col(history, "search_kl"),
        "s_dent": col(history, "search_dentropy"),
    }


def _mark_interventions(ax, key):
    """Draw a vertical dashed line + rotated caption at each registered intervention
    that targets this panel (panels=="all" or key in panels). Label rides the top of
    the axes via the xaxis transform so it never collides with the data."""
    for iv in INTERVENTIONS:
        panels = iv.get("panels", "all")
        if panels != "all" and key not in panels:
            continue
        x = iv["iter"]
        ax.axvline(x, color=C["marker"], ls=(0, (4, 3)), lw=1.2, alpha=0.7, zorder=1.5)
        # Put the label on whichever side of the line has more room (left of the line when it sits in
        # the right third of the x-range, else right), and give it an opaque white box so it reads as
        # a clean tag over data/gridlines/annotations instead of tangling with them.
        xmin, xmax = ax.get_xlim()
        on_right = x > xmin + 0.6 * (xmax - xmin)
        ax.text(x, 0.985, iv["label"], transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha=("right" if on_right else "left"),
                fontsize=6, color=C["marker"], zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C["marker"],
                          lw=0.5, alpha=0.9))


def _int_xaxis(ax, it, mark=None):
    # Adaptive tick base: every-5 labels in the normal 10–60-iter range, every iteration
    # for short fresh-restart runs (a <5-iter run otherwise shows ONLY the '0' label),
    # scaled up in steps of 5 for long runs so labels never crowd (span 300 → base 25).
    span = int(np.max(it)) - int(np.min(it)) if len(it) else 0
    base = 1 if span < 10 else 5 * max(1, round(span / 60))
    ax.xaxis.set_major_locator(MultipleLocator(base))
    if base > 1:
        ax.xaxis.set_minor_locator(MultipleLocator(max(1, base // 5)))
    ax.set_xlabel("Iteration")
    # Intervention marker lines disabled — they were overlapping/covering the data. The registry
    # (INTERVENTIONS) + _mark_interventions() are kept dormant; re-enable by restoring this call.
    _ = mark


def _legend_below(ax, ax2=None, ncol=2):
    """Place the legend below the axis (clear of the data and the x-label),
    merging a twin axis's handles into one legend so they never stack/overlap."""
    h, lab = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, lab2 = ax2.get_legend_handles_labels()
        h, lab = h + h2, lab + lab2
    # Anchor well below the x-axis label so the legend never overlaps the data
    # or the "Iteration" caption.
    ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=ncol, frameon=False)


# ----------------------------------------------------------------------------
# Panels — each draws onto a supplied axis so the dashboard and the standalone
# figures share exactly one implementation.
# ----------------------------------------------------------------------------

def panel_elo(ax, H):
    it, elo, ci, meas = H["it"], H["elo"], H["elo_ci"], H["elo_measured"]
    mask = ~np.isnan(elo)
    ax.plot(it[mask], elo[mask], color=C["elo"], lw=2.2, zorder=2,
            label="Champion Elo (vs Stockfish)")
    # Iteration 0 = the supervised-pretrained base (distinct marker + annotation);
    # exclude it from the regular RL "Measured" scatter below.
    is_pre = it == 0
    if is_pre.any():
        i0 = np.where(is_pre)[0][0]
        ax.scatter([0], [elo[i0]], marker="D", s=90, color=C["value"], zorder=5,
                   edgecolor="white", linewidth=1.2, label="Pretrained (Lichess-1800 SL)")
        sf0 = "" if np.isnan(H["sf_elo"][i0]) else f" · vs SF {int(H['sf_elo'][i0])}"
        ax.annotate(f"Lichess-1800 SL\npretrain{sf0}", (0, elo[i0]),
                    textcoords="offset points", xytext=(10, 4), fontsize=7,
                    color=C["value"], fontweight="bold")
    # Measured points (filled, with CI bars where available) vs carried-forward (hollow).
    m_meas = mask & meas & ~is_pre
    m_carry = mask & ~meas & ~is_pre
    cm = ~np.isnan(ci) & m_meas
    if cm.any():
        ax.errorbar(it[cm], elo[cm], yerr=ci[cm], fmt="none", ecolor=C["elo"],
                    elinewidth=1.4, capsize=4, alpha=0.7, zorder=1)
    ax.scatter(it[m_meas], elo[m_meas], s=55, color=C["elo"], zorder=3,
               edgecolor="white", linewidth=1, label="Measured")
    if m_carry.any():
        ax.scatter(it[m_carry], elo[m_carry], s=45, facecolor="white",
                   edgecolor=C["elo"], linewidth=1.6, zorder=3, label="Carried forward")
    # Promotion stars.
    prom = mask & H["promoted"]
    if prom.any():
        ax.scatter(it[prom], elo[prom] + 6, marker="*", s=130, color=C["champ"],
                   zorder=4, label="Promoted")
    # Annotate Stockfish anchor at each measured point.
    for i in np.where(m_meas)[0]:
        if not np.isnan(H["sf_elo"][i]):
            ax.annotate(f"SF {int(H['sf_elo'][i])}", (it[i], elo[i]),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=7, color="#64748b")
    ax.set_title("Elo Progression")
    ax.set_ylabel("Elo")
    _int_xaxis(ax, it, "elo")
    _legend_below(ax, ncol=4)


def panel_loss(ax, H, which):
    it = H["it"]
    if which == "policy":
        tr, va, c, name = H["p_loss"], H["vp_loss"], C["policy"], "Policy"
    else:
        tr, va, c, name = H["v_loss"], H["vv_loss"], C["value"], "Value"
    mt, mv = ~np.isnan(tr), ~np.isnan(va)
    ax.plot(it[mt], tr[mt], color=c, lw=2.2, marker="o", ms=4, label=f"{name} train")
    ax.plot(it[mv], va[mv], color=c, lw=1.8, ls="--", marker="s", ms=3.5,
            alpha=0.75, label=f"{name} val")
    # Shade the generalization gap where both exist.
    both = mt & mv
    if both.any():
        ax.fill_between(it[both], tr[both], va[both], color=c, alpha=0.10)
    ax.set_title(f"{name} Loss (train vs val)")
    ax.set_ylabel("Loss")
    _int_xaxis(ax, it, "loss")
    _legend_below(ax, ncol=2)


def panel_acc(ax, H):
    it = H["it"]
    mt, mv = ~np.isnan(H["train_acc"]), ~np.isnan(H["val_acc"])
    ax.plot(it[mt], H["train_acc"][mt], color=C["policy"], lw=2.2, marker="o",
            ms=4, label="Train acc")
    ax.plot(it[mv], H["val_acc"][mv], color=C["value"], lw=2.2, marker="s",
            ms=4, label="Val acc")
    ax.set_title("Policy Top-1 Accuracy")
    ax.set_ylabel("Accuracy (%)")
    _int_xaxis(ax, it, "acc")
    _legend_below(ax, ncol=2)


def panel_kl(ax, H):
    it = H["it"]
    m = ~np.isnan(H["kl_anchor"])
    ax.plot(it[m], H["kl_anchor"][m], color=C["kl"], lw=2.2, marker="o", ms=4,
            label="KL(candidate ‖ pretrained prior)")
    ax.set_title("KL-to-Reference  (anchor, β=1.0)")
    ax.set_ylabel("KL (nats)")
    ax.set_ylim(bottom=0)
    _int_xaxis(ax, it, "kl")
    _legend_below(ax, ncol=1)


def panel_search(ax, H):
    it = H["it"]
    ax.plot(it, H["s_top1"], color=C["search"], lw=2.0, marker="o", ms=4,
            label="net==MCTS top1 (%)")
    ax.plot(it, H["s_override"], color=C["winrate"], lw=2.0, marker="s", ms=4,
            label="MCTS override (%)")
    ax.set_title("Search vs Net (policy improvement)")
    ax.set_ylabel("%")
    _int_xaxis(ax, it, "search")
    # KL(MCTS‖net) on a twin axis — different scale (nats).
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(it, H["s_kl"], color=C["cand"], lw=1.8, ls="--", marker="^", ms=3.5,
             label="KL(MCTS ‖ net)")
    ax2.set_ylabel("KL (nats)", color=C["cand"])
    ax2.tick_params(axis="y", labelcolor=C["cand"])
    _legend_below(ax, ax2, ncol=3)


def panel_value_calib(ax, H):
    it = H["it"]
    ax.plot(it, H["va_cand"], color=C["cand"], lw=2.2, marker="o", ms=4,
            label="Value acc — candidate (%)")
    ax.plot(it, H["va_champ"], color=C["champ"], lw=2.0, ls="--", marker="s", ms=3.5,
            label="Value acc — champion (%)")
    ax.set_title("Value Head Calibration")
    ax.set_ylabel("Accuracy (%)")
    _int_xaxis(ax, it, "value")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(it, H["dec_cand"], color=C["value"], lw=1.6, ls=":", marker="d", ms=3.5,
             label="Decisive |v|≥0.9 (%)")
    ax2.set_ylabel("Decisive (%)", color=C["value"])
    ax2.tick_params(axis="y", labelcolor=C["value"])
    _legend_below(ax, ax2, ncol=3)


def panel_grad(ax, H):
    it = H["it"]
    m = ~np.isnan(H["grad_norm"])
    ax.plot(it[m], H["grad_norm"][m], color=C["grad"], lw=2.2, marker="o", ms=4,
            label="Grad norm")
    ax.set_title("Gradient Norm (stability)")
    ax.set_ylabel("‖grad‖")
    ax.set_ylim(bottom=0)
    _int_xaxis(ax, it, "grad")
    _legend_below(ax, ncol=1)


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------

def build_dashboard(H):
    # Elo is the sole progress metric (arena panels removed) → it headlines the top row
    # spanning the full width; the training + probe panels fill the two rows below.
    fig = plt.figure(figsize=(24, 20), dpi=DPI)
    gs = GridSpec(3, 3, figure=fig, hspace=0.78, wspace=0.34,
                  left=0.05, right=0.97, top=0.93, bottom=0.04)
    panel_elo(fig.add_subplot(gs[0, :]), H)
    panel_loss(fig.add_subplot(gs[1, 0]), H, "policy")
    panel_loss(fig.add_subplot(gs[1, 1]), H, "value")
    panel_acc(fig.add_subplot(gs[1, 2]), H)
    panel_kl(fig.add_subplot(gs[2, 0]), H)
    panel_search(fig.add_subplot(gs[2, 1]), H)
    panel_value_calib(fig.add_subplot(gs[2, 2]), H)

    # Headline + footer.
    elo_now = H["elo"][~np.isnan(H["elo"])]
    champ = int(elo_now[-1]) if elo_now.size else None
    n_prom = int(H["promoted"].sum())
    title = "Self-Play RL — Training Dashboard"
    if champ is not None:
        title += f"   •   champion ≈ {champ} Elo   •   {n_prom} promotions / {len(H['it'])} iters logged"
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.99, 0.005, f"generated {datetime.now():%Y-%m-%d %H:%M}",
             ha="right", fontsize=8, color="#94a3b8")
    out = os.path.join(OUTPUT_DIR, "dashboard.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  dashboard      → {out}")


def save_panel(name, drawfn, H, figsize=(10, 6.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    drawfn(ax, H)
    out = os.path.join(OUTPUT_DIR, name)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name:<22} → {out}")


def main():
    setup_style()
    history = load_metrics()
    if not history:
        print("Metrics file is empty.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Prepend the pretrained base as iteration 0 so every panel includes it
    # wherever a comparable stat exists (Elo, accuracy, KL-anchor).
    if history[0].get("iteration") != 0:
        history = [PRETRAIN_RECORD] + history
    H = parse(history)

    print(f"Plotting {len(history)} iterations ({H['it'][0]}–{H['it'][-1]})...")
    build_dashboard(H)

    # Standalone panels. Elo is the headline progress metric; the rest are the
    # training + offline-probe panels. (Arena win-rate / reward-KL removed — the
    # Stockfish-anchored gate replaced the arena.)
    save_panel("elo_rating.png", panel_elo, H)
    save_panel("policy_loss.png", lambda ax, h: panel_loss(ax, h, "policy"), H)
    save_panel("validation_loss.png", lambda ax, h: panel_loss(ax, h, "value"), H)
    save_panel("accuracy.png", panel_acc, H)
    save_panel("kl_anchor.png", panel_kl, H)
    save_panel("search_improvement.png", panel_search, H)
    save_panel("value_calibration.png", panel_value_calib, H)
    save_panel("grad_norm.png", panel_grad, H)

    print("\nDone.")


if __name__ == "__main__":
    main()
