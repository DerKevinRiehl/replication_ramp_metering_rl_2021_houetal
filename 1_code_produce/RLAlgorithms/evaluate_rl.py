import argparse
import csv
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RL_ALGO_DIR = PROJECT_ROOT / "RLAlgorithms"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RL_ALGO_DIR))

from environment.env import RampMeterEnv
from routes.generate_routes import generate_extreme_flows, write_route_file
from routes.generate_routes import generate_mixed_flows, write_route_file

#paths are analogous to run_baselines.py

ROUTES_DIR = PROJECT_ROOT / "routes"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.sumocfg"
RESULTS_DIR = PROJECT_ROOT / "results"

#analogous to run_one_baseline_episode.py: generate per-episode route + config files
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

#---- spatial speed pipeline ----
SPACE_SPEED_DIR = PROJECT_ROOT / "results" / "space_speed"
#-----------------------------------------


def load_trained_algorithm(checkpoint_path: str):
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # kill all rollout workers
    algo.workers.foreach_worker(lambda w: w.stop())
    algo.workers._workers = []  

    return algo

def run_episode(algo, episode_idx, algo_name, traffic_mode, config_path):

    #here we do one evaluation episode with the trained RL - agent (create environment, call reset(), run episode, extract final metrics)

    #---- spatial speed pipeline ----
    #One .npz per (mode, algo, episode). Enabled for RL evaluation.
    space_speed_path = SPACE_SPEED_DIR / traffic_mode / f"{algo_name}_ep{episode_idx:02d}.npz"
    #-----------------------------------------

    env = RampMeterEnv(
        sumo_binary="sumo",
        seed=episode_idx,
        config_path=str(config_path),
        dynamic_routes=False, #use config and routes that were generated here, do not create a dynamic one in env like in training
        collect_space_speed=True,                     # speed plotting pipeline: enable per-step spatial speed logging
        space_speed_path=str(space_speed_path),
    )

    state, _ = env.reset()

    done = False
    info = {}

    while not done:
        
        if env.step_count < env.warmup_steps: #skip warm up, hence dummy value (traffic light will be green during the warm up phase)
            action = -1
        else:
            # explore=False: in evaluation phase want to choose the learned policy deterministically (no more epsilon exploration)
            
            action = algo.compute_single_action(
                observation=state,
                explore=False
            )

            #extract action if it is a tuple (depends on RLlib version)
            if isinstance(action, tuple):
                action = action[0]

            # then convert action to an int (we want 0 or 1)
            action = int(action)
        
        state, reward, done, truncated, info = env.step(action)

    env.close()
    return info


def run_experiment(algo, algo_name: str, episodes: int, traffic_mode: str):

    rows = []

    ROUTES_GEN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_GEN_DIR.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        print(f"Running {algo_name} - episode {episode}")

        #episodes should be reproducable, but not identical to each other
        rng = np.random.default_rng(episode)

        if traffic_mode == "extreme":
            main, ramp = generate_extreme_flows(rng)
        else:
            main, ramp = generate_mixed_flows(rng)

        route_path = ROUTES_GEN_DIR / f"routes_{algo_name}_ep{episode:02d}.rou.xml"
        config_path = CONFIG_GEN_DIR / f"config_{algo_name}_ep{episode:02d}.sumocfg"

        write_route_file(route_path, main, ramp)
        make_episode_config(BASE_CONFIG_PATH, config_path, route_path)

        metrics = run_episode(algo, episode, algo_name, traffic_mode, config_path)
        metrics["controller"] = algo_name
        metrics["episode"] = episode
        metrics["mode"] = traffic_mode

        rows.append(metrics)

    return rows


def save_results(rows, output_path: Path):
    
    #save data in csv file 
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No result rows available to save.")

    fieldnames = rows[0].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    
    parser = argparse.ArgumentParser(
        description="Evaluate one trained RL algorithm on the ramp metering environment."
    )

    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "apexdqn", "a3c"],
        help="name of RL-Algorithm"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib-Checkpoint of trained modell"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="how many evaluation episodes should be done?"
    )

    parser.add_argument(
        "--traffic",
        type=str,
        default="mixed",
        choices=["extreme", "mixed"],
        help="which traffic version should be used? (mixed / extreme)"
    )

    args = parser.parse_args() #parse input 

    #save directly into the per_episode folder with mode in filename
    #(matches the location and naming plot_paper_figures.py expects)
    output_path = RESULTS_DIR / "per_episode" / f"{args.algo}_{args.traffic}.csv"

    ray.init(local_mode=True, num_cpus=1)

    from ray import tune
    from function_utils import create_environment

    tune.register_env("ramp_meter_environment", create_environment)

    algo = load_trained_algorithm(args.checkpoint)

    

    rows = run_experiment(
        algo=algo,
        algo_name=args.algo,
        episodes=args.episodes,
        traffic_mode=args.traffic
    )

    save_results(rows, output_path)

    print(f"Finished evaluation for {args.algo}")
    print(f"Results saved to: {output_path}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
