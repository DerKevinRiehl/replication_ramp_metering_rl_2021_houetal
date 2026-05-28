from pathlib import Path 
import pandas as pd 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_EP_DIR = PROJECT_ROOT / "results" / "per_episode"
MERGED_DIR = PROJECT_ROOT / "results" / "merged"
MERGED_DIR.mkdir(parents = True, exist_ok=True)

files = sorted(RESULTS_EP_DIR.glob("result_*.csv"))
if not files:
    raise ValueError("no per-episode result files found")

dfs = []

for f in files: #read in all the different data frames (results from each episode) into a list 
    data = pd.read_csv(f)
    dfs.append(data)

df = pd.concat(dfs, ignore_index = True) #then concat them into one big data frame 

df.to_csv(MERGED_DIR / "baseline_results_all.csv", index = False) #make a csv file out of it 


summary = df.groupby(["mode", "controller"]).agg({
    "throughput_veh_h": ["mean", "std"],
    "speed_m_s": ["mean", "std"], 
    "stops_per_vehicle": ["mean", "std"], 
    "fuel_mpg": ["mean", "std"], 
    "co2_g_mi": ["mean", "std"], 
    "nox_mg_mi": ["mean", "std"], 
}).reset_index()


summary.to_csv(MERGED_DIR / "baseline_results_summary.csv", index=False)

print("merged results saved")
print(summary)



