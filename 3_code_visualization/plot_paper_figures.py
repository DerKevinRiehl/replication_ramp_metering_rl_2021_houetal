"""
Reproduce Figures 4 and 5 from Hou et al. —-> ramp meter signal control paper.

Figure 4: Comparison in mixed near-congested and congested traffic conditions
Figure 5: Comparison in extremely congested traffic conditions
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Anchor all paths to the project root so the script works regardless of CWD
PROJECT_ROOT    = Path(__file__).resolve().parents[1]
PER_EPISODE_DIR = PROJECT_ROOT / "results" / "per_episode"
OUTPUT_DIR      = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# canonical controller order (used everywhere in boxplots, legends, colors)
CONTROLLER_ORDER = ["PPO", "Ape-X DQN", "A3C", "ALINEA", "No RM", "Fixed"]

# legend display labels (paper writes "Fixed (Red=3s)")
CONTROLLER_DISPLAY = {
    "PPO":       "PPO",
    "Ape-X DQN": "Ape-X DQN",
    "A3C":       "A3C",
    "ALINEA":    "ALINEA",
    "No RM":     "No RM",
    "Fixed":     "Fixed (Red=3s)",
}

# fill colors — chosen to match the paper figure's palette
CONTROLLER_COLORS = {
    "PPO":       "#3B7DD8",   # strong blue
    "Ape-X DQN": "#5BB85B",   # grass green
    "A3C":       "#E8514B",   # red
    "ALINEA":    "#E0B22E",   # golden yellow
    "No RM":     "#48BFC4",   # petrol
    "Fixed":     "#C896E0",   # lilac
}

# robust mapping from any raw spelling to canonical name
CONTROLLER_ALIASES = {
    "ppo":          "PPO",
    "apex":         "Ape-X DQN",
    "apexdqn":      "Ape-X DQN",
    "ape-xdqn":     "Ape-X DQN",
    "ape_x_dqn":    "Ape-X DQN",
    "a3c":          "A3C",
    "alinea":       "ALINEA",
    "norm":         "No RM",
    "no_rm":        "No RM",
    "norampmeter":  "No RM",
    "fixed":        "Fixed",
    "fixedtime":    "Fixed",
    "fixed_time":   "Fixed",
}

# metric definitions 
METRICS = [
    {"col": "throughput_veh_h",  "ylabel": "Traffic Throughput (veh/h)",  "tag": "(a)"},
    {"col": "speed_m_s",         "ylabel": "Vehicle speed (m/s)",         "tag": "(b)"},
    {"col": "stops_per_vehicle", "ylabel": "Number of Stops per Vehicle", "tag": "(c)"},
    {"col": "fuel_mpg",          "ylabel": "Fuel Efficiency (mpg)",       "tag": "(d)"},
    {"col": "co2_g_mi",          "ylabel": "CO2 (g/mi)",                  "tag": "(e)"},
    {"col": "nox_mg_mi",         "ylabel": "NOx (mg/mi)",                 "tag": "(f)"},
]



def _slug(s):

    #Lower-case, strip separators 
    return "".join(c for c in str(s).lower() if c.isalnum())


def normalize_controller(name):

    #Map any raw controller name to one of the canonical labels 
    
    if pd.isna(name):
        return None
    s = _slug(name)
    if s in {_slug(k): k for k in CONTROLLER_ORDER}.values():

        
        for canon in CONTROLLER_ORDER:
            if _slug(canon) == s:
                return canon
    if s in CONTROLLER_ALIASES:
        return CONTROLLER_ALIASES[s]
    return str(name) 


def detect_mode_from_filename(filename):

    #Detect 'mixed' or 'extreme' from filename
    
    lower = filename.lower()
    if "extreme" in lower:
        return "extreme"
    if "mixed" in lower:
        return "mixed"
    return None


def detect_controller_from_filename(filename):

    s = _slug(filename)
    for alias, canon in CONTROLLER_ALIASES.items():
        if alias in s:
            return canon
    for canon in CONTROLLER_ORDER:
        if _slug(canon) in s:
            return canon
    return None


def load_per_episode_data(per_episode_dir):
    
    #Load and concatenate all per-episode CSVs in the directory.

    csv_files = sorted(Path(per_episode_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in '{per_episode_dir}/'")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)

       
        if "controller" not in df.columns:
            ctrl = detect_controller_from_filename(path.name)
            if ctrl is None:
                print(f"[warn] cannot detect controller for {path.name}")
                continue
            df["controller"] = ctrl

        
        if "mode" not in df.columns:
            mode = detect_mode_from_filename(path.name)
            if mode is None:
                print(f"[warn] cannot detect mode for {path.name}")
                continue
            df["mode"] = mode

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["controller"] = full["controller"].apply(normalize_controller)
    full["mode"]       = full["mode"].astype(str).str.lower().str.strip()
    return full


def setup_ggplot_style():

    
    plt.rcParams.update({
        "font.family":         "DejaVu Sans",
        "font.size":            9,
        "axes.facecolor":      "#EBEBEB",
        "axes.edgecolor":      "white",
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "axes.labelcolor":     "#222222",
        "grid.color":          "white",
        "grid.linewidth":       0.7,
        "xtick.color":         "#555555",
        "ytick.color":         "#555555",
        "xtick.major.size":     0,
        "ytick.major.size":     0,
        "legend.frameon":       False,
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
    })


def draw_boxplot_panel(ax, df_mode, metric_col, ylabel, present):
    

    data, colors = [], []
    for ctrl in present:
        values = df_mode[df_mode["controller"] == ctrl][metric_col].dropna().values
        data.append(values)
        colors.append(CONTROLLER_COLORS[ctrl])

    positions = list(range(1, len(data) + 1))

    bp = ax.boxplot(
        data,
        positions    = positions,
        widths       = 0.55,
        patch_artist = True,
        showmeans    = True,
        whis         = 1.5,      
        meanprops    = dict(marker="D", markerfacecolor="white",
                            markeredgecolor="black", markersize=5,
                            markeredgewidth=0.7),
        medianprops  = dict(color="black", linewidth=1.0),
        whiskerprops = dict(color="black", linewidth=0.8),
        capprops     = dict(color="black", linewidth=0.8),
        flierprops   = dict(marker="o", markerfacecolor="none",
                            markeredgecolor="black", markersize=3.5,
                            markeredgewidth=0.6),
    )

    # color each box separately
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)
        patch.set_alpha(0.85)

    # color outliers in their controller's color (paper style)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_alpha(0.9)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks([])  
    ax.set_xlim(0.4, len(data) + 0.6)


def add_panel_legend(ax, present):

    
    handles = [
        mpatches.Patch(facecolor=CONTROLLER_COLORS[c], edgecolor="black",
                       linewidth=0.5, label=CONTROLLER_DISPLAY[c])
        for c in present
    ]
    ax.legend(
        handles         = handles,
        loc             = "center left",
        bbox_to_anchor  = (1.02, 0.5),
        fontsize        = 7,
        handlelength    = 1.0,
        handleheight    = 1.0,
        labelspacing    = 0.45,
        borderaxespad   = 0.0,
    )


def make_figure(df, mode, output_stem):
    

    sub = df[df["mode"] == mode].copy()
    if sub.empty:
        return

    present = [c for c in CONTROLLER_ORDER if c in sub["controller"].unique()]
    if not present:
        return

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.5, 7.5))

    for idx, metric in enumerate(METRICS):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        draw_boxplot_panel(ax, sub, metric["col"], metric["ylabel"], present)
        add_panel_legend(ax, present)
        ax.text(0.5, -0.14, metric["tag"], transform=ax.transAxes,
                ha="center", va="top", fontsize=10)

    fig.subplots_adjust(left=0.06, right=0.95, top=0.97, bottom=0.07,
                        wspace=0.95, hspace=0.42)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,            bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")



def main():
    setup_ggplot_style()
    df = load_per_episode_data(PER_EPISODE_DIR)

    print(f"\nLoaded {len(df)} episode rows")
    print(f"Controllers : {sorted(df['controller'].dropna().unique().tolist())}")
    print(f"Modes       : {sorted(df['mode'].dropna().unique().tolist())}\n")

    make_figure(df, mode="mixed",   output_stem=OUTPUT_DIR / "figure4_mixed")
    make_figure(df, mode="extreme", output_stem=OUTPUT_DIR / "figure5_extreme")


if __name__ == "__main__":
    main()
