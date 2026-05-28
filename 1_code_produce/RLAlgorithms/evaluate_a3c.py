


#seperate evaluation file for A3C, as loading the algorithm checkpoint creates issue due to asynchronous workers in A3C. 
#Hence we only load the policy itself here to solve this problem 


import sys
print(sys.executable)
import ray
print(ray.__file__)

import argparse
import csv
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RL_ALGO_DIR = PROJECT_ROOT / "RLAlgorithms"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RL_ALGO_DIR))


from environment.env import RampMeterEnv
from routes.generate_routes import (
    generate_extreme_flows,
    generate_mixed_flows,
    write_route_file,
)

from ray.rllib.policy.policy import Policy

ROUTES_DIR = PROJECT_ROOT / "routes"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.sumocfg"
RESULTS_DIR = PROJECT_ROOT / "results"

#----spatial speed pipeline ----
SPACE_SPEED_DIR = PROJECT_ROOT / "results" / "space_speed"
#-------------------------------

#analogous to evaluate_rl.py / run_one_baseline_episode.py
BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.sumocfg"
ROUTES_GEN_DIR = PROJECT_ROOT / "routes" / "generated"
CONFIG_GEN_DIR = PROJECT_ROOT / "config" / "generated"


def make_episode_config(base_config_path, out_config_path, route_file_path):
    net_path = PROJECT_ROOT / "network" / "ramp.net.xml"
    veh_types = PROJECT_ROOT / "routes" / "vehicle_types.xml"

    new_config = f"""
<configuration>

<input>
    <net-file value="{net_path}" />
    <route-files value="{veh_types},{route_file_path}" />
</input>

<time>
    <begin value="0" />
    <end value="1040" />
    <step-length value="1" />
</time>

<processing>
    <seed value="42" />
    <time-to-teleport value="-1"/>
</processing>

</configuration>
"""

    Path(out_config_path).write_text(new_config)


# Load Policy (NO RLlib ALGO, NO WORKERS)

def load_policy(checkpoint_path: str):
    print("loading policy")
    policy = Policy.from_checkpoint(checkpoint_path)
    print("policy is loaded")

    # FIX: extract actual policy
    if isinstance(policy, dict):
        policy = policy["default_policy"]

    return policy



#run single episode 

def run_episode(policy, episode_idx: int, config_path, traffic_mode: str):

    #----spatial speed pipeline ----
    space_speed_path = SPACE_SPEED_DIR / traffic_mode / f"a3c_ep{episode_idx:02d}.npz"
    #-------------------------------

    env = RampMeterEnv(
        sumo_binary="sumo",
        config_path=str(config_path),
        dynamic_routes=False,
        seed=episode_idx,
        algo_name="a3c_eval",
        collect_space_speed=True,
        space_speed_path=str(space_speed_path),
    )

    state, _ = env.reset()

    done = False
    info = {}

    while not done:

        if env.step_count < env.warmup_steps:
            action = -1
        else:
            action = policy.compute_single_action(state)

            if isinstance(action, tuple):
                action = action[0]

            action = int(action)


        state, reward, done, truncated, info = env.step(action)

    env.close()
    return info



#experiment loop 

def run_experiment(policy, episodes: int, traffic_mode: str):

    rows = []

    ROUTES_GEN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_GEN_DIR.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        print(f"Running A3C - episode {episode}")

        rng = np.random.default_rng(episode)

        if traffic_mode == "extreme":
            main, ramp = generate_extreme_flows(rng)
        else:
            main, ramp = generate_mixed_flows(rng)

        route_path = ROUTES_GEN_DIR / f"routes_a3c_ep{episode:02d}.rou.xml"
        config_path = CONFIG_GEN_DIR / f"config_a3c_ep{episode:02d}.sumocfg"

        write_route_file(route_path, main, ramp)
        make_episode_config(BASE_CONFIG_PATH, config_path, route_path)

        metrics = run_episode(policy, episode, config_path, traffic_mode)

        metrics["controller"] = "a3c"
        metrics["episode"] = episode
        metrics["mode"] = traffic_mode

        rows.append(metrics)

    return rows


#save all results 

def save_results(rows, output_path: Path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No results to save.")

    fieldnames = rows[0].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)



def main():

    print("starting main method")

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--traffic", type=str, default="mixed", choices=["mixed", "extreme"])

    args = parser.parse_args()

    #save directly into the per_episode folder with mode in filename
    #(matches the location and naming plot_paper_figures.py expects)
    output_path = RESULTS_DIR / "per_episode" / f"a3c_{args.traffic}.csv"

    print("callling load_policy")

    policy = load_policy(args.checkpoint)

    rows = run_experiment(
        policy=policy,
        episodes=args.episodes,
        traffic_mode=args.traffic
    )

    save_results(rows, output_path)

    print("Finished A3C evaluation")
    print(f"Saved to: {output_path}")



if __name__ == "__main__":
    main()