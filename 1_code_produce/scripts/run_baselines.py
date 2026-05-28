
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from environment.env import RampMeterEnv
from controllers.no_meter import NoMeterController
from controllers.fixed_time import FixedTimeController
from controllers.alinea import AlineaController
from routes.generate_routes import generate_mixed_flows, generate_extreme_flows, write_route_file


ROUTES_DIR = PROJECT_ROOT / "routes"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.sumocfg"
RESULT_PATH = PROJECT_ROOT / "results" / "baseline_results.csv"

#---- spatial speed pipeline ----
SPACE_SPEED_DIR = PROJECT_ROOT / "results" / "space_speed"
TRAFFIC_MODE = "mixed"  # must match the flow generator used below
#-------------------------

def make_controller(name):

    if name == "noRM":
        return NoMeterController()
    
    if name == "Fixed":
        return FixedTimeController()
    
    if name == "Alinea":
        return AlineaController()
    
    raise ValueError("invalid controller")




def run_episode(controller, seedEpisode, controller_name):

    #---- Figure 6 spatial speed pipeline ----
    space_speed_path = SPACE_SPEED_DIR / TRAFFIC_MODE / f"{controller_name}_ep{seedEpisode:02d}.npz"
    #-----------------------------------------

    env = RampMeterEnv(
        sumo_binary="sumo",                  # CHANGE HERE to sumo-gui for visualization
        seed=seedEpisode,
        dynamic_routes=False,
        config_path=str(CONFIG_PATH),
        collect_space_speed=True,            
        space_speed_path=str(space_speed_path),
    )

    state, _ = env.reset()
    controller.reset()

    done = False 
    info = {}

    while not done:

        if env.step_count < env.warmup_steps:
            action = -1 #dummy value, environment will skip action execution until warmup is done  
        else:
            if isinstance(controller, AlineaController):
                action = controller.compute_action(state, env = env)
            else: 
                action = controller.compute_action(state) #NoRM and FixedTimeRM don't need current traffic information

        state, reward, done, truncated, info = env.step(action)

    env.close()
    return info 



def run_experiment():

    controllers = [
        "noRM",
        # "Fixed",
        # "Alinea",
    ]

    rows = []

    for contr_name in controllers:
        
        print("--------------------------------")
        print("Runnig", contr_name)
        print("--------------------------------")

        for episode in range(20): #change amount of episodes here for different testing purposes 

            rng = np.random.default_rng(episode) #important: every episode we choose a new seed, otherwise we always get the same random numbers, BUT the seeds are equal for each controller (per episode)

            main, ramp = generate_mixed_flows(rng) #HERE YOU CAN SWITCH between mixed and extreme flows 

            write_route_file(ROUTES_DIR / "routes_mixed.rou.xml", main, ramp) #SWITCH FILE NAME HERE ASWELL 

            controller = make_controller(contr_name)

            metrics = run_episode(controller, episode, contr_name)
            metrics["controller"] = contr_name
            metrics["episode"] = episode

            rows.append(metrics)
    
    return rows



def save_results(rows):

    fieldnames = rows[0].keys()

    with open(RESULT_PATH, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)


if __name__ == "__main__": 

    rows = run_experiment()

    save_results(rows)

    print("Finished baseline experiments")


