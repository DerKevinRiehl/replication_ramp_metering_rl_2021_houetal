#in this file helper functions used from all 3 RL algorithms are stored for modularity and clean code 


#creates the environment instance for the RL algorithm 
def create_environment(env_config):

    import sys, os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)

    from environment.env import RampMeterEnv #lazy loading, module is only loaded if function is called (performance improvement)

    cfg = dict(env_config)
    cfg["worker_index"] = getattr(env_config, "worker_index", 0)
    return RampMeterEnv(**cfg) #return instance of environment



#Robust extraction of relevant RLlib metrics 
def extract_training_metrics(result: dict, iteration: int) -> dict:
    
    clean = {
        "iteration": iteration,
        "training_iteration": result.get("training_iteration"),
        "timesteps_total": result.get("timesteps_total"),
        "episodes_total": result.get("episodes_total"),
        "episode_reward_mean": result.get("episode_reward_mean"),
    }

    custom_metrics = result.get("custom_metrics", {})

    clean["throughput_veh_h"] = custom_metrics.get("throughput_veh_h_mean")
    clean["speed_m_s"] = custom_metrics.get("speed_m_s_mean")
    clean["total_speed_m_s"] = custom_metrics.get("total_speed_m_s_mean")

    return clean
