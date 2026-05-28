#computes mean metric values per RL algorithm and across all 3 RL algorithms together
#reads from results/per_episode/{algo}_{mode}.csv (the output of evaluate_rl.py / evaluate_a3c.py)
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_EPISODE_DIR = PROJECT_ROOT / "results" / "per_episode"
RESULTS_DIR = PROJECT_ROOT / "results"

RL_ALGOS = ["ppo", "apexdqn", "a3c"]
MODE = "mixed"  #change to "extreme" if needed

METRICS = ["throughput_veh_h", "speed_m_s", "stops_per_vehicle", "fuel_mpg", "co2_g_mi", "nox_mg_mi"]


def find_csv(algo, mode):
    #try new naming first (per_episode/{algo}_{mode}.csv)
    p = PER_EPISODE_DIR / f"{algo}_{mode}.csv"
    if p.exists():
        return p

    #fallback to old naming (results/{algo}_evaluation_results.csv)
    p = RESULTS_DIR / f"{algo}_evaluation_results.csv"
    if p.exists():
        return p
    return None


def main():

    #------load all 3 RL csvs and stack them together------
    frames = []
    for algo in RL_ALGOS:
        csv_path = find_csv(algo, MODE)
        if csv_path is None:
            print(f"[warn] no csv found for {algo}")
            continue
        df = pd.read_csv(csv_path)
        frames.append(df)

    if not frames:
        print("no RL csvs found, nothing to compute")
        return

    all_rl = pd.concat(frames, ignore_index=True)
    

    #------per-algorithm mean values-----
    print(f"\n=== Mean values per RL algorithm (mode={MODE}) ===\n")
    for algo in RL_ALGOS:
        sub = all_rl[all_rl["controller"] == algo]
        if sub.empty:
            continue
        print(f"--- {algo} ({len(sub)} episodes) ---")
        for metric in METRICS:
            mean = sub[metric].mean()
            print(f"  {metric}: {mean:.4f}")
        print()
    

    #------combined mean across all RL algorithms----------
    print(f"=== Combined mean across all RL algorithms ({len(all_rl)} episodes) ===\n")
    for metric in METRICS:
        mean = all_rl[metric].mean()
        print(f"  {metric}: {mean:.4f}")
    print()
    #------------------------------------------------------


if __name__ == "__main__":
    main()
