
import os #standard library for operating system functions (for absolute path to write results to, no matter which machine)
import json #for formating of logged data 
import sys
import shutil 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

for p in [PROJECT_ROOT, RL_ALGO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- logging fix ---
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "1"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_AUTOSCALER_V2"] = "0"
os.environ["RAY_verbose_spill_logs"] = "0"

os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR-NUM_THREADS"] = "1"


import ray #framework for distributed computing (we need it here for the actors in the algorithm )
from ray import tune 
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig #config builder for the algorihtm (which env, how many actors, learning rate, ... )

from early_stopping import RewardPlateauEarlyStopping
from function_utils import extract_training_metrics
from function_utils import create_environment

from tqdm import tqdm 
from live_plot import LivePlot 
from callbacks import RampMeterCallbacks

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from pathlib import Path


if __name__ == "__main__": 
    
    ray.init(num_cpus=32, ignore_reinit_error=True, log_to_driver=False, include_dashboard=False, logging_level="ERROR", configure_logging=False)

    tune.register_env("ramp_meter_environment", create_environment)

    output_directory = os.path.join(PROJECT_ROOT, "RLAlgorithms", "apexdqn_runs")
    os.makedirs(output_directory, exist_ok=True)


    config = (

        ApexDQNConfig()

        .environment(
            env = "ramp_meter_environment",
            env_config = {
                "sumo_binary": "sumo",
                "config_path": os.path.join(PROJECT_ROOT, "config", "config.sumocfg"),
                "algo_name": "apexdqn", #dynamic routes are true as the default in env
            },
        )
        .callbacks(RampMeterCallbacks)
        .framework("torch")
        .resources(num_gpus = 0)

        .rollouts(num_rollout_workers = 32,
                  rollout_fragment_length = 1000, 
                  recreate_failed_workers = True, 
                  restart_failed_sub_environments = True,) #use 32 workers and 32 cpus

        .training(
            gamma = 0.99, #paper values 
            lr = 5e-4,
            train_batch_size = 1000,

            model = {"fcnet_hiddens": [128], "fcnet_activation": "tanh"},

            replay_buffer_config = {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 5_000_000,
            },
        )

        .exploration(
            exploration_config = {
                "type": "EpsilonGreedy", 
                "initial_epsilon": 0.1,
                "final_epsilon": 1e-4 ,
            }
        )
    )

    algo = config.build()
    result = {"timesteps_total": 0}

    target_timesteps = 40_000_000
    prev_timesteps = 0

    logging_path = os.path.join(output_directory, "train_log_apexdqn.jsonl")

    #------------checkpoint logic------------------
    checkpoint_interval = 1_000_000
    last_checkpoint = 0
    max_checkpoints = 5
    checkpoint_paths = []
    best_reward = -float("inf")
    best_checkpoint_path = None
    best_dir = Path(output_directory) / "best_checkpoint"
    #----------------------------------------------

    early_stopper = RewardPlateauEarlyStopping(min_iterations=30, patience = 15, min_delta=0.5, smoothing_window=5)

    progress_bar = tqdm(total = target_timesteps, desc = "APEXDQN Training", unit = "ts", dynamic_ncols=True, position = 0, leave=True)
    live_plot = LivePlot(save_path=os.path.join(output_directory, "apexdqn_live_reward.png"), algo_name="apexdqn")

    iteration = 0 

    while result.get("timesteps_total", 0) < target_timesteps:
        
        result = algo.train()

        clean_result = extract_training_metrics(result, iteration)

        current_timesteps = clean_result.get("timesteps_total", 0)

        #--------live plotting------------
        reward = clean_result.get("episode_reward_mean")
        timesteps = clean_result.get("timesteps_total")

        if reward is not None:
            live_plot.update(timesteps, reward)
        #---------------------------

        #-------tqdm update---------------------------
        delta = current_timesteps - prev_timesteps
        if delta > 0:
            progress_bar.update(delta)
        prev_timesteps = current_timesteps

        progress_bar.set_postfix({
            "reward": round(clean_result.get("episode_reward_mean", 0), 2) if clean_result.get("episode_reward_mean") else None,
            "speed": round(clean_result.get("speed_m_s", 0), 2) if clean_result.get("speed_m_s") else None,
            "throughput": round(clean_result.get("throughput_veh_h", 0), 2) if clean_result.get("throughput_veh_h") else None,
            "episodes": clean_result.get("episodes_total") if clean_result.get("episodes_total") else None,
        })
        #---------------------------------------------

        early_stopping = early_stopper.update(clean_result.get("episode_reward_mean"))

        #---------------optinal logging-----------------
        #if iteration % 60 == 0:
        #    with open(logging_path, "a") as f:
        #        f.write(json.dumps(clean_result, default=str) + "\n")
        #--------------------------------------

        #-----------------checkpoint-------------------------
        if current_timesteps - last_checkpoint >= checkpoint_interval:

            checkpoint_path = algo.save(output_directory)
            checkpoint_paths.append(checkpoint_path)
            print(f"Checkpoint path saved at iteration {iteration}: {checkpoint_path}")

            if len(checkpoint_paths) > max_checkpoints:
                old_path = checkpoint_paths.pop(0)
                shutil.rmtree(old_path, ignore_errors=True)
            
            last_checkpoint = current_timesteps
        
        if reward is not None and reward > best_reward + 5:

            best_reward = reward

            if best_dir.exists():
                shutil.rmtree(best_dir)

            if checkpoint_paths:
                checkpoint_path = checkpoint_paths[-1]
            else:
                checkpoint_path = algo.save(output_directory)
                checkpoint_paths.append(checkpoint_path)

            if checkpoint_path is not None:
                shutil.copytree(checkpoint_path, best_dir)
                best_checkpoint_path = str(best_dir)
        #-------------------------------------------------- 

        #----------optional early stopping--------
        #if early_stopping:
        #    print("EARLY STOPPING TRIGGERED AS REWARD IS PLATEAUING, BUT WE KEEP TRAINING")
        #-----------------------------------------

        iteration += 1
    
    progress_bar.close()

    checkpoint_path = algo.save(output_directory)
    print("final checkpoint: ", checkpoint_path)

    ray.shutdown()