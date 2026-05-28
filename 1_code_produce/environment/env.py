import traci #controls sumo (traci is the python API we use to control SUMO)
import subprocess 
import sys 
import numpy as np
import os
import gymnasium as gym 
from gymnasium import spaces
import random 

from routes.generate_routes import generate_mixed_flows, generate_extreme_flows, write_route_file
from pathlib import Path #for RL training, such that every episode gets its own temporary route file 

#---- spatial speed pipeline for speed plot ----
#Helper that collects per-step vehicle speeds in spatial bins (5m) along the mainline.
#Imported here at the top so the optional logger can be constructed in reset()
from environment.space_speed_logger import SpaceSpeedLogger
#-----------------------------------------


class RampMeterEnv(gym.Env):

    
    def __init__(self, sumo_binary = "sumo", seed=42, config_path = None, dynamic_routes = True, worker_index = 0, algo_name = "rl",
                 collect_space_speed = False, space_speed_path = None):

        self.sumo_binary = sumo_binary
        self.algo_name = algo_name

        #-------RL Training -------------
        self.base_seed = seed #base_seed is the starting seed
        self.episode_idx = 0 #which episode are we currently in
        self.dynamic_routes = dynamic_routes #with this we can switch between baseline logic and rl logic (route generation for baseline logic is in run_one_basleine_episode and not in env, because we run it manually, in RL training we rely on Rllib and hence overwrite the route and config files here in env)
        self.base_config_path = config_path 
        self.generated_route_path = None
        self.generated_config_path = None

        self.project_root = Path(__file__).resolve().parent.parent
        self.generated_routes_dir = (self.project_root / "routes" / "generated_rl" / algo_name).resolve()
        self.generated_configs_dir = (self.project_root / "config" / "generated_rl" / algo_name).resolve()
        self.generated_routes_dir.mkdir(parents=True, exist_ok=True)
        self.generated_configs_dir.mkdir(parents=True, exist_ok=True)

        self.worker_index = worker_index
        self.episode_seed = 0
        #---------------------------------

        if config_path is None:
            raise ValueError("config_path must be provided") 

        self.sumo_cmd = [self.sumo_binary, "-c", self.base_config_path, "--seed", str(self.base_seed), "--no-warnings", "--no-step-log"] 

        self.tl_id = "node11" #traffic light is located at node11 of the network

        self.step_count = 0 #needed to skip warm up in metrics 
        self.max_step_count = 1000 #for each simulation episode, 1000 time steps with each time step as one second (paper)
        self.warmup_steps = 40 #paper says we do 40 warm up steps each episode 

        #------speed attributes---------
        self.bottleneck_speed_acc = 0 #measure average speed locally on a specific segment 
        self.bottleneck_speed_vehicle_counter = 0
        self.bottleneck_local_speed_measurement = "seg6" #merging area

        self.total_speed_sum = 0 #measure average speed globally in the whole network 
        self.total_speed_count = 0
        #------------------------------

        self.segments = ["seg1", "seg2", "seg3", "seg4", "seg5", "seg6", "seg7", "seg8", "seg9", "seg10"] #due to definition of network (paper)

        self.previous_action = 0 #needed for state

        self.speed_max = 29.0576 #highway speed (65mph) in m/s 

        self.throughput_buffer_state = [0] * 30 #last entry of state requires the throughput of the last 30 seconds

        self.action_space = spaces.Discrete(2) #traffic light has only 2 states: red or green 
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(40,), dtype=np.float32) #state has 40 dimensions (see paper)

        #assumptions 
        self.gasoline_density_g_per_ml = 0.745 #needed to convert fuel mass in grams to volume in ml for mpg 
        self.ml_per_gallon = 3785.411784 #needed to convert fuel volume in ml to gallons for mpg 
        self.m_per_mile = 1609.344 #needed to convert distance meters --> miles for g/mi, mg/mi, mpg
        self.stop_speed_threshold = 1 #defines a vehicle being "stopped" if speed < threshold in m/s, used for stop-event counting

        self.freeway_entry_edge = "seg1"
        self.ramp_entry_edge = "seg9"
        self.bottleneck_out_edge = "seg7" #for measuring throughput for actual metric
        self.throughput_state_out_edge = "seg7" #for measuring throughput for the state 

        #edge id memory for flow counting 
        self.prev_freeway_edge_ids = set()
        self.prev_ramp_edge_ids = set()
        self.prev_bottleneck_edge_ids = set()
        self.prev_throughput_state_edge_ids = set()
       
        self._reset_metrics() #initialize all accumulators in the beginning

        #---- spatial speed pipeline for speed illustration across whole network ----
        #Off by default so training is never slowed down. Enabled by the
        #evaluation scripts (run_one_baseline_episode.py and evaluate_rl.py),
        #which also pass the output path. A fresh logger is created in
        #reset() each episode and data is written out at episode end in step().
        self.collect_space_speed = collect_space_speed
        self.space_speed_path = space_speed_path
        self.space_speed_logger = None
        #-----------------------------------------

    def _reset_metrics(self): #keeps reset() clean and seperate from metrics

        self.control_time_s = 0.0 #needed for avg speed + throughput normalization, ONLY post warm up 
        self.total_arrived_network = 0 #counts arrived vehicles over episode for throughput in veh/h 
        self.total_bottleneck_out = 0 #bottleneck outflow count 
        self.total_distance_m = 0.0 #distance driven across all vehicles for per - mile metrics 
        self.total_fuel_mg = 0.0 #total fuel consumption per episode 
        self.total_co2_mg = 0.0 #total co2 emissions per episode
        self.total_nox_mg = 0.0 #total nox emissions per episode 
        self.total_vehicle_time = 0 #how long a vehicle is present

        #for speed measurement in bottleneck 
        self.bottleneck_speed_acc = 0
        self.bottleneck_speed_vehicle_counter = 0

        #for speed measurement in whole network 
        self.total_speed_sum = 0 
        self.total_speed_count = 0

        self.stop_events = 0 #counter for transitions to "stopped" state
        self.seen_vehicle_ids = set() #unique vehicles per episode (for denominator in stops per vehicle)
        self.prev_speed = {} #remembers last speed per vehicle to detect stopping

        #per-vehicle accumulators for "per vehicle" metrics (CO2, NOx)
        #paper labels these explicitly as "Average XY emissions per vehicle"
        #hence we compute per-vehicle ratio (for example co2_i / miles_i), then mean across vehicles
        self.per_vehicle_distance_m = {}  #vid -> meters
        self.per_vehicle_co2_mg = {}       #vid -> mg
        self.per_vehicle_nox_mg = {}       #vid -> mg

        #---------last measured per-step and per 15 sec inflows for alinea------------

        #default setting here 
        self.freeway_in_buffer = [0] * 1 #change this to * 10 for tuned parameters to smooth measurements 
        self.ramp_in_buffer = [0] * 1
        
        #-----------------------------------------------------------------------------

    

    #helper to count new vehicles on an edge, this tells us how many vehicles entered/passed that specific edge
    def _count_new_ids_on_edge(self, edge_id, prev_ids):
        
        current_ids = set(traci.edge.getLastStepVehicleIDs(edge_id))
        new_ids = current_ids - prev_ids #how many of the current ids are actually "new" on the edge?
        return len(new_ids), current_ids

        
    #only gets called when warm up (40 steps) is done
    def _update_metrics_step(self):  # handles per step accumulation, such that step() stays readable and compact 

        dt_s = traci.simulation.getDeltaT() #length of one simulation step in seconds

        # --- not needed during RL training (ALINEA controller only), remove comments for evaluation ---
        freeway_in_count, self.prev_freeway_edge_ids = self._count_new_ids_on_edge(self.freeway_entry_edge, self.prev_freeway_edge_ids)
        ramp_in_count, self.prev_ramp_edge_ids = self._count_new_ids_on_edge(self.ramp_entry_edge, self.prev_ramp_edge_ids)
        # ---------------------------------------------------------------

        bottleneck_out_count, self.prev_bottleneck_edge_ids = self._count_new_ids_on_edge(self.bottleneck_out_edge, self.prev_bottleneck_edge_ids)

        throughput_state_count, self.prev_throughput_state_edge_ids = self._count_new_ids_on_edge(self.throughput_state_out_edge, self.prev_throughput_state_edge_ids)

        #----Alinea variables, remove comments for evaluation --------------------- not needed during RL training

        self.freeway_in_buffer.pop(0)
        self.freeway_in_buffer.append(freeway_in_count)

        self.ramp_in_buffer.pop(0)
        self.ramp_in_buffer.append(ramp_in_count)
    
        #--------------------------------------------

        self.throughput_buffer_state.pop(0)
        self.throughput_buffer_state.append(throughput_state_count)

        self.control_time_s += dt_s  # we track total episode time for throughput and avg-speed

        # --- not needed during RL training, remove comments for evaluation  ---
        arrived = traci.simulation.getArrivedNumber()  #vehicles that finished this step
        self.total_arrived_network += arrived
        # -------------------------------------
        self.total_bottleneck_out += bottleneck_out_count

        #------Speed Measurement local--------
        vehicle_ids_bottleneck = traci.edge.getLastStepVehicleIDs(self.bottleneck_local_speed_measurement)
        for vid in vehicle_ids_bottleneck:

            self.bottleneck_speed_acc += traci.vehicle.getSpeed(vid)
            self.bottleneck_speed_vehicle_counter += 1
        #-------------------------------------

        # --- not needed during RL training (no global speed / emission / stop metrics), remove comments for evaluation  ---
        vehicle_ids = traci.vehicle.getIDList()  # all vehicles currently in simulation for distance/emission computations

        #-----Speed Measurement global--------
        for vid in vehicle_ids:
            self.total_speed_sum += traci.vehicle.getSpeed(vid)
            self.total_speed_count += 1
        #-------------------------------------

        current_set = set(vehicle_ids)  #used to clean up prev_speed dict so it doesn't get too large, we use a set here to optimize performance (lookup in a set is O(1) due to hashes instead of O(n) in the tuple here)

        self.total_vehicle_time += len(vehicle_ids) * dt_s

        for vid in vehicle_ids:  # integrate single vehicle contributions into episode total accumulators
            v = traci.vehicle.getSpeed(vid)  #speed in m/s within the last step
            self.total_distance_m += v * dt_s  #distance = speed * time, (dt_s = 1, just for clarity) sum across all vehicle, as SUMO returns rates per second, (we do multiply with dt_s for clarity, but keepp in mind dt_s = 1 here)
            self.total_fuel_mg += traci.vehicle.getFuelConsumption(vid) * dt_s # fuel consumption in mg/s
            co2_step = traci.vehicle.getCO2Emission(vid) * dt_s  #CO2 emission in mg/s
            nox_step = traci.vehicle.getNOxEmission(vid) * dt_s  #NOx emission in mg/s
            self.total_co2_mg += co2_step
            self.total_nox_mg += nox_step #per-vehicle accumulation for paper-faithful "per vehicle" CO2 and NOx
            self.per_vehicle_distance_m[vid] = self.per_vehicle_distance_m.get(vid, 0.0) + v * dt_s
            self.per_vehicle_co2_mg[vid]     = self.per_vehicle_co2_mg.get(vid, 0.0) + co2_step
            self.per_vehicle_nox_mg[vid]     = self.per_vehicle_nox_mg.get(vid, 0.0) + nox_step
            self.seen_vehicle_ids.add(vid)  # count unique vehicles for stops-per-vehicle normalization
            prev_v = self.prev_speed.get(vid, None)  # last speed to detect transitions into stopping state
            if prev_v is not None:  # only count transitions if we have a previous speed
                if prev_v >= self.stop_speed_threshold and v < self.stop_speed_threshold: #vehicle was driving but now it stopped
                    self.stop_events += 1  #thus count one stop event
            self.prev_speed[vid] = v  #update speed for next step's transition detection

        ## here we remove vehicles that left the simulation so prev_speed doesn't accumulate stale IDs
        for old_vid in list(self.prev_speed.keys()):  #iterate over tracked IDs
            if old_vid not in current_set:
                self.prev_speed.pop(old_vid, None)  # delete entry if vehicle doesn't exist anymore
        # ---------------------------------------------------------------------------------


    def _finalize_metrics_info(self):  #computes unit metrics they use in the paper, at end of episode/simulation

        throughput_veh_h = self.total_bottleneck_out / max(self.control_time_s, 1e-9) * 3600.0 #scaling to convert to hours 
        
        avg_speed_m_s_bottleneck = self.bottleneck_speed_acc / max(self.bottleneck_speed_vehicle_counter, 1e-9) #already in m/s

        avg_speed_m_s_total = self.total_speed_sum / max(self.total_speed_count, 1e-9) 

        n_unique = max(len(self.seen_vehicle_ids), 1)  #avoiding divide-by-zero if no vehicles were seen
        stops_per_vehicle = self.stop_events / n_unique  #stopping-events normalized by number of unique vehicles

        miles = self.total_distance_m / self.m_per_mile  # convert total distance from meters into miles

        #per-vehicle CO2 and NOx (paper: "Average CO2/NOx emissions per vehicle")
        #compute g/mi (or mg/mi) for each vehicle individually, then mean across vehicles
        co2_per_vehicle_values = []
        nox_per_vehicle_values = []
        for vid, dist_m in self.per_vehicle_distance_m.items():
            miles_i = dist_m / self.m_per_mile
            if miles_i < 1e-6:
                continue  #skip vehicles that essentially never moved (no meaningful ratio)
            co2_g_i = self.per_vehicle_co2_mg.get(vid, 0.0) / 1000.0
            nox_mg_i = self.per_vehicle_nox_mg.get(vid, 0.0)
            co2_per_vehicle_values.append(co2_g_i / miles_i)
            nox_per_vehicle_values.append(nox_mg_i / miles_i)

        co2_g_per_mi = sum(co2_per_vehicle_values) / max(len(co2_per_vehicle_values), 1)
        nox_mg_per_mi = sum(nox_per_vehicle_values) / max(len(nox_per_vehicle_values), 1)

        fuel_g = self.total_fuel_mg / 1000.0  # mg to g describing the mass of fuel 
        fuel_ml = fuel_g / self.gasoline_density_g_per_ml  #g to ml using density (assumption documented in init)
        fuel_gal = fuel_ml / self.ml_per_gallon  #ml to gallons for mpg
        mpg = miles / max(fuel_gal, 1e-12)  # miles per gallon

        return {  
            "throughput_veh_h": throughput_veh_h,  
            "speed_m_s": avg_speed_m_s_bottleneck, 
            #"total_speed_m_s" : avg_speed_m_s_total,
            "stops_per_vehicle": stops_per_vehicle,  
            "fuel_mpg": mpg,  
            "co2_g_mi": co2_g_per_mi,  
            "nox_mg_mi": nox_mg_per_mi, 

            #metrics for debugging and better network understanding 

            #"bottleneck_out_total": self.total_bottleneck_out, 
            #"network_arrived_total": self.total_arrived_network, 
            #"unique_vehicles": len(self.seen_vehicle_ids), 
            #"control_time_s": self.control_time_s,
        }



    #RL Training 

    #creates new route file and temporary sumocfg pointing to it (as we have multiple workers in the RL algorithms, overwriting the same 
    #config file would most likely result in data races, hence we need to have seperate configs for each worker and for each episode)
    #only used when dynamic_routes = True 

    #gets called by each worker for each episode in training 
    def _build_episode_specific_sumo_cmd(self):
        
        print("creating dynamic routes")
        
        #-------clean up (otherwise memory will be used up soon) ---------

        #example: worker A did first episode, then it sees file 1 & 2 already exist, hence delete it and make new ones for next episode (all other workers remain untouched)
        #intuition: each worker only needs the most recent config and route  file, older ones can be deleted to save memory
        if self.generated_route_path and os.path.exists(self.generated_route_path):
            os.remove(self.generated_route_path)
        
        if self.generated_config_path and os.path.exists(self.generated_config_path):
            os.remove(self.generated_config_path)
        #-----------------------

        # unique seed for each (worker, episode) combination — without the
        # worker_index offset every parallel worker would generate the exact
        # same routes for the same episode_idx (since base_seed is identical
        # across workers per episode), which kills experience diversity during training.
        episode_seed = self.base_seed + self.episode_idx + self.worker_index * 100_000

        # worker id form RLlib 
        worker_idx = getattr(self, "worker_index", "main")

        # unique file names per Algo + Worker + Episode + Seed to avoid data races
        route_path = self.generated_routes_dir / f"routes_{self.algo_name}_worker{worker_idx}_ep{self.episode_idx}_seed{episode_seed}.rou.xml"
        cfg_path = self.generated_configs_dir / f"config_{self.algo_name}_worker{worker_idx}_ep{self.episode_idx}_seed{episode_seed}.sumocfg"

        rng = np.random.default_rng(episode_seed)
        mixed_flows, ramp_flows = generate_mixed_flows(rng)
        write_route_file(route_path, mixed_flows, ramp_flows)

        with open(self.base_config_path, "r") as f:
            cfg_text = f.read()


        # --- build clean sumo config with correct seed and route files ---

        network_path = (self.project_root / "network" / "ramp.net.xml").resolve()
        vehicle_types_path = (self.project_root / "routes" / "vehicle_types.xml").resolve()

        cfg_text = f"""<configuration>
            <input>
                <net-file value="{network_path}" />
                <route-files value="{vehicle_types_path},{route_path}" />
            </input>

            <time>
                <begin value="0"/>
                <end value="1040"/>
                <step-length value="1"/>
            </time>

            <processing>
                <seed value="{episode_seed}"/>
                <time-to-teleport value="-1"/>
            </processing>
        </configuration>
        """

        with open(cfg_path, "w") as f:
            f.write(cfg_text)

        self.generated_route_path = str(route_path)
        self.generated_config_path = str(cfg_path)

        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            self.generated_config_path,
            "--seed",
            str(episode_seed), #new seed here for each episode and worker
            "--no-warnings",
            "--no-step-log",
        ]



    def reset(self, *, seed=None, options=None): #starts a new simulation, gets called at the beginning of each episode

        if traci.isLoaded():
            traci.close() #otherwise we might get a problem if we call start() if sumo is still running 
        
        #--------------RL Training ----------------
        if self.dynamic_routes:
            self._build_episode_specific_sumo_cmd() #create config and route file for workers
            self.episode_idx += 1
        #------------------------------------------

        label = f"worker_{self.worker_index}"

        traci.start(self.sumo_cmd, label=label) #start new simulation with command store in attribute sumo_cmd (for rl training the command was overwritten in build function)
        traci.switch(label)

        self.step_count = 0 #reset counter
        self.previous_action = 0 #reset prev action flag 

        self.throughput_buffer_state = [0] * 30 #reset buffer too 

        self.prev_freeway_edge_ids = set()
        self.prev_ramp_edge_ids = set()
        self.prev_bottleneck_edge_ids = set()
        self.prev_throughput_state_edge_ids = set()

        self._reset_metrics() #reset all metrics at the start of each episode

        #---- Figure 6 spatial speed pipeline ----
        #Build a fresh logger per episode when collection is enabled.
        if self.collect_space_speed:
            self.space_speed_logger = SpaceSpeedLogger()
        else:
            self.space_speed_logger = None
        #-----------------------------------------

        state = self._get_state()
        assert state.shape == (40,)
        return state, {} #return state, st agent can choose his first action


    def step(self, action):

        #apply action 
       
        if action == 0:
            traci.trafficlight.setRedYellowGreenState(self.tl_id, "r") #set traffic light to red 
        else:
            traci.trafficlight.setRedYellowGreenState(self.tl_id, "G") #set traffic light to green 
            
        self.previous_action = action

        #execute one simulation step 
        traci.simulationStep()
        self.step_count += 1 #increase step count 

        #only track metrics if warm up is done 
        state = self._get_state() #state always needs to exist
        reward = 0
        if(self.step_count >= self.warmup_steps):

            self._update_metrics_step() #update all the metrics for one step

            reward = self._get_reward()

            #---- spatial speed pipeline ----
            #Add this step's vehicle speeds to the per-bin accumulators.
            if self.space_speed_logger is not None:
                self.space_speed_logger.record_step()
            #-----------------------------------------

        done = self.step_count >= (self.max_step_count + self.warmup_steps)

        terminated = done 
        truncated = False
        
        info = {}
        if terminated:
            info = self._finalize_metrics_info() #if finished, return final metrics

            #---- spatial speed pipeline ----
            #Persist this episode's per-bin accumulators to disk.
            #The plotting script later aggregates these per controller.
            if self.space_speed_logger is not None and self.space_speed_path is not None:
                self.space_speed_logger.save(self.space_speed_path)
            #-----------------------------------------
            
        return state, reward, terminated, truncated, info

        

    def _get_state(self):

        state = []

        for seg in self.segments: #iterate trough all segments 

            lanes = traci.edge.getLaneNumber(seg) #get number of lanes from current segment

            for curr_lane in range(lanes):

                curr_lane_id = f"{seg}_{curr_lane}" #for example: seg6_1

                speed = traci.lane.getLastStepMeanSpeed(curr_lane_id) #mean speed of vehicles on that lane 
                count = traci.lane.getLastStepVehicleNumber(curr_lane_id) #number of vehicles on that lane
                

                state.append(speed)
                state.append(count) 
        
        state.append(self.previous_action) #citation from paper: "ramp meter signal phase (green or red) of the previous simulation step"


        throughput_30seconds = sum(self.throughput_buffer_state) #last entry is the total throughput of the last 30 steps 
        state.append(throughput_30seconds)

        return np.array(state, dtype=np.float32)
                

    #implements the reward function given in the paper
    def _get_reward(self):

        v6 = traci.edge.getLastStepMeanSpeed("seg6")
        v9 = traci.edge.getLastStepMeanSpeed("seg9")
        v10 = traci.edge.getLastStepMeanSpeed("seg10")

        v_bar = (v6 + v9 + v10) / 3 

        speed_term = v_bar / self.speed_max #normalize speed 

        cm = 0.0
        seg6_ids = traci.edge.getLastStepVehicleIDs("seg6")
        seg6_slow = 0
        for v_id in seg6_ids:
            if traci.vehicle.getSpeed(v_id) < 1.0:
                seg6_slow += 1
        
        if seg6_slow >= 5:
            cm = -1.0
        
        cr = 0.0
        seg9_ids = traci.edge.getLastStepVehicleIDs("seg9")
        seg9_slow = 0
        for v_id in seg9_ids:
            if traci.vehicle.getSpeed(v_id) < 1:
                seg9_slow += 1 
        
        if seg9_slow >= 9:
            cr = -1.0 
        

        return float (speed_term + cm + cr)
        

    #helper for alinea 
    def get_last_measured_flows_vph(self):

        window_s = len(self.freeway_in_buffer)

        return {

            "q_freeway_vph": sum(self.freeway_in_buffer) / window_s * 3600.0, 
            "q_ramp_vph": sum(self.ramp_in_buffer) / window_s * 3600.0, 
        }

    #helper for alinea 
    def _count_segment_vehicles(self, edge_id):
        return len(traci.edge.getLastStepVehicleIDs(edge_id))

    
    def close(self): #closes the simulation 

        label = f"worker_{self.worker_index}"

        try:
            traci.switch(label)
            traci.close()
        except:
            pass