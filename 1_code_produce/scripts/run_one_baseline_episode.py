import argparse
import csv
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from environment.env import RampMeterEnv
from controllers.no_meter import NoMeterController
from controllers.alinea import AlineaController
from controllers.fixed_time import FixedTimeController
from routes.generate_routes import generate_mixed_flows, generate_extreme_flows, write_route_file

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.sumocfg"
ROUTES_GEN_DIR = PROJECT_ROOT / "routes" / "generated"
CONFIG_GEN_DIR = PROJECT_ROOT / "config" / "generated"
RESULTS_EP_DIR = PROJECT_ROOT / "results" / "per_episode"

#---- spatial speed pipeline ----
SPACE_SPEED_DIR = PROJECT_ROOT / "results" / "space_speed"


def make_controller(name):

    if name == "noRM":
        return NoMeterController()
    
    if name == "Fixed":
        return FixedTimeController()
    
    if name == "Alinea":
        return AlineaController()
    
    raise ValueError("invalid controller in make_controller")


#make config for each episode due to the different routes generation 

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

def run_episode(controller_name, episode, mode):

    ROUTES_GEN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_GEN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_EP_DIR.mkdir(parents= True, exist_ok = True)

    rng = np.random.default_rng(episode) #use episode index as seed 

    if mode == "mixed":
        main, ramp = generate_mixed_flows(rng)
    elif mode == "extreme":
        main, ramp = generate_extreme_flows(rng)
    else:
        raise ValueError("mode must be mixed or extreme for flow generation")
    
    route_path = ROUTES_GEN_DIR / f"routes_{controller_name}_ep{episode:02d}.rou.xml"
    config_path = CONFIG_GEN_DIR / f"config_{controller_name}_ep{episode:02d}.sumocfg"
    result_path = RESULTS_EP_DIR / f"result_{controller_name}_ep{episode:02d}.csv"

    write_route_file(route_path, main, ramp)
    make_episode_config(BASE_CONFIG_PATH, config_path, route_path)

    #---- spatial speed pipeline ----
    space_speed_path = SPACE_SPEED_DIR / mode / f"{controller_name}_ep{episode:02d}.npz"
    #-----------------------------------------

    env = RampMeterEnv(
        sumo_binary="sumo",
        seed=episode,
        config_path=str(config_path),
        dynamic_routes = False, 
        collect_space_speed=True,           
        space_speed_path=str(space_speed_path),
    )

    controller = make_controller(controller_name)
    
    state, _ = env.reset()
    controller.reset()

    done = False 
    info = {}

    while not done:

        if env.step_count < env.warmup_steps:
            action = -1 #choose dummy action here, st traffic light is always green 
        else: 

            if controller_name == "Alinea":
                action = controller.compute_action(state, env = env)
            else:
                action = controller.compute_action(state)
        
        state, reward, done, truncated, info = env.step(action)
    
    env.close()

    info["controller"] = controller_name
    info["episode"] = episode
    info["mode"] = mode

    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=info.keys())
        writer.writeheader()
        writer.writerow(info)

    print(f"finished {controller_name} episode {episode}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(())

    parser.add_argument("--controller", required = True, choices = ["noRM", "Fixed", "Alinea"])

    parser.add_argument("--episode", required = True, type = int)

    parser.add_argument("--mode", default = "mixed", choices = ["mixed", "extreme"])

    args = parser.parse_args()

    run_episode(args.controller, args.episode, args.mode)
    




