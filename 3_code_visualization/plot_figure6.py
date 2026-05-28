"""
Reproduce Figure 6 from Hou et al. --> spatial speed heatmap.

One row per controller, x-axis = longitudinal coordinate along the freeway
mainline, color = average vehicle speed at that location averaged over all
evaluation episodes and all simulation steps. A red rectangle marks the
merging area (seg6).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
SPACE_SPEED_DIR = PROJECT_ROOT / "results" / "space_speed"
OUTPUT_DIR      = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Top-to-bottom row order in the figure. Strings must match the filename
# prefixes produced by the evaluation scripts (case-sensitive).
CONTROLLER_ORDER = ["ppo", "apexdqn", "a3c", "Alinea", "noRM", "Fixed"]

# Displaying labels on the y-axis (paper style)
CONTROLLER_DISPLAY = {
    "ppo":     "PPO",
    "apexdqn": "Ape-X DQN",
    "a3c":     "A3C",
    "Alinea":  "ALINEA",
    "noRM":    "No RM",
    "Fixed":   "Fixed (Red=3s)",
}

# Speed color range for the heatmap (m/s). -> Maatches paper's scale (0..30).
SPEED_VMIN = 0.0
SPEED_VMAX = 30.0



def load_controller_data(mode_dir, controller):

    # Aggregate sum_v and cnt across all episodes for one controller.

    files = sorted(mode_dir.glob(f"{controller}_ep*.npz"))
    if not files:
        return None

    total_sum = None
    total_cnt = None
    bin_edges = None
    merge_x_start = None
    merge_x_end = None

    for f in files:
        d = np.load(f)
        # sum_v and cnt are already 1D per episode — hence just add them up across episodes
        s = d["sum_v"]
        c = d["cnt"]

        if total_sum is None:
            total_sum = s.astype(np.float64)
            total_cnt = c.astype(np.int64)
            bin_edges = d["bin_edges"]
            merge_x_start = float(d["merge_x_start"])
            merge_x_end   = float(d["merge_x_end"])
        else:
            total_sum += s
            total_cnt += c

    # vehicle-weighted mean per bin of bins that never saw a vehicle become NaN
    mean_speed = np.where(
        total_cnt > 0,
        total_sum / np.maximum(total_cnt, 1),
        np.nan,
    )
    return mean_speed, bin_edges, merge_x_start, merge_x_end

#plotting istelf 
def make_figure(mode):
    
    #Build and save one heatmap figure for a given traffic mode (mixed/extreme)
    
    mode_dir = SPACE_SPEED_DIR / mode
    if not mode_dir.exists():
        print(f"[warn] no data directory for mode='{mode}'")
        return

    rows = []
    labels = []
    bin_edges = None
    merge_x_start = None
    merge_x_end = None

    for ctrl in CONTROLLER_ORDER:
        result = load_controller_data(mode_dir, ctrl)
        if result is None:
            print(f"[warn] no episodes for '{ctrl}' in mode '{mode}'")
            continue
        mean_speed, bin_edges, merge_x_start, merge_x_end = result
        rows.append(mean_speed)
        labels.append(CONTROLLER_DISPLAY[ctrl])

    if not rows:
        print(f"[warn] nothing to plot for mode='{mode}'")
        return

    speeds = np.stack(rows, axis=0)  # (n_controllers, n_bins)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    # extent = (x_left, x_right, y_bottom, y_top) as rows are stacked top to bottom
    extent = (float(bin_edges[0]), float(bin_edges[-1]), len(labels), 0)
    im = ax.imshow(
        speeds,
        aspect="auto",
        cmap="viridis",                # dark-blue -> green -> yellow  (matches paper)
        extent=extent,
        vmin=SPEED_VMIN,
        vmax=SPEED_VMAX,
        interpolation="bilinear",      # smooth color transitions between bins
    )

    # one tick per controller, centered on its row
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Distance (m)")

    # horizontal white separator lines between controller rows
    # in the paper style — each controller row is visually distinct
    for y in range(1, len(labels)):
        ax.axhline(y, color="white", linewidth=2.5, zorder=3)

    # red rectangle marking the merging area (seg6)
    rect = mpatches.Rectangle(
        (merge_x_start, 0),
        merge_x_end - merge_x_start,
        len(labels),
        linewidth=2.0,
        edgecolor="red",
        facecolor="none",
        zorder=4,
    )
    ax.add_patch(rect)

    fig.colorbar(im, ax=ax, label="Speed (m/s)")
    fig.tight_layout()

    png_path = OUTPUT_DIR / f"figure6_{mode}.png"
    pdf_path = OUTPUT_DIR / f"figure6_{mode}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")




def main():
    make_figure("mixed")
    make_figure("extreme")


if __name__ == "__main__":
    main()
