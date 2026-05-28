import numpy as np 
from pathlib import Path 

ROUTES_DIR = Path("routes") #save generated route files in "routes"
ROUTES_DIR.mkdir(exist_ok = True)

def clip(x, lower_bound, upper_bound):

    #paper wants the drawn values to be clipped (page 3)
    return max(lower_bound, min(upper_bound, x))


def generate_mixed_flows(rng):

    #generates flows for the "mixed traffic" scenario 

    main_flows = []
    ramp_flows = []

    for _ in range(10):

        q_main = rng.normal(3000, 300) #freeway flow ~ N(3000, 300)
        q_ramp = rng.normal(900, 300) #ramp flow ~ N(900, 300)

        q_main = int(round(clip(q_main, 0, 3900))) #proceed with clipping the values like descriped in the paper (page 3)
        q_ramp = int(round(clip(q_ramp, 0, 1500)))

        main_flows.append(q_main)
        ramp_flows.append(q_ramp)

    return main_flows, ramp_flows



def generate_extreme_flows(rng):

    #generates flows for the extremely congested scenario 

    main_flows = [3900] * 10 #paper: "freeway inflow was set as constant 3900 vph"
    ramp_flows = [] #same as in mixed flows for ramp inflow 

    for _ in range(10): 

        q_ramp = rng.normal(900, 300) # q_ramp ~ N(900, 300)
        q_ramp = int(round(clip(q_ramp, 0, 1500))) #clip values 
        ramp_flows.append(q_ramp)
    
    return main_flows, ramp_flows


def write_route_file(filepath, main_flows, ramp_flows):  # turn the collected flows into route files with right formatting

    lines = ["<routes>"]
    for i in range(len(main_flows)):

        begin = i * 100  # create the intervals
        end = (i + 1) * 100

        q_main = int(main_flows[i])
        if q_main > 0:
            lines.append(
                f'<flow id="main_{i}" begin="{begin}" end="{end}" '
                f'vehsPerHour="{q_main}" from="seg1" to="seg8" '
                f'departSpeed="max" departLane="best" departPos="free" '
                f'insertionChecks="none" type="car"/>'
            )
        q_ramp = int(ramp_flows[i])
        if q_ramp > 0:
            lines.append(
                f'<flow id="ramp_{i}" begin="{begin}" end="{end}" '
                f'vehsPerHour="{q_ramp}" from="seg9" to="seg8" '
                f'departSpeed="max" departLane="best" departPos="free" '
                f'insertionChecks="none" type="car"/>'
            )

    lines.append("</routes>")

    filepath.write_text("\n".join(lines))


if __name__ == "__main__":

    rng = np.random.default_rng(42) #keep seed fixed for reproducible outcomes

    #generate mixed scenario 
    main_flows, ramp_flows = generate_mixed_flows(rng)
    write_route_file(ROUTES_DIR / "routes.mixed.rou.xml", main_flows, ramp_flows)

    #generate extreme scenario 
    main_flows, ramp_flows = generate_extreme_flows(rng)
    write_route_file(ROUTES_DIR / "routes_extreme.rou.xml", main_flows, ramp_flows)


