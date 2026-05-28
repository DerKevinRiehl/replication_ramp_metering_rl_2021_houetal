import traci


class AlineaController:

    #ALINEA baseline controller, cycle time gets re-evaluated every 15 seconds, 1 sec green, rest red 


    def __init__(self, ramp_lanes=1, Cf=20, n_star=10, update_interval=15): #these values are defined in the paper

        self.N = ramp_lanes
        self.Cf = Cf
        self.n_star = n_star
        self.update_interval = update_interval

        self.t = 0 
        self.N = 1 #number of ramp lanes

        #temporary default cycle here, until we compute the cycle time after 15 s
        #we choose cycle_time = 4 as this corresponds to the fixed-time cycle (see file fixed_time.py)
        self.cycle_time = 4
        self.red_time = 3

        self.prev_n = None 
        self.prev_q_total = None 

        self.timer = 0

    def reset(self):

        #here we reset the cycle, thus we again choose the same values as in the fixed-time cycle (same as in init)
        self.t = 0
        self.timer = 0
        self.cycle_time = 4
        self.red_time = 3

        self.prev_n = None 
        self.prev_q_total = None 


    #here we pass env explicitly so alinea can use the measured inflows (also smoothed inflows over last 10 simulation steps, se tuned metrics in paper)
    def compute_action(self, state=None, env=None):

        if env is None:
            raise ValueError("alinea requires env")

        # --- update cycle time every 15s --- 
        if self.t % self.update_interval == 0 and self.t > 0:

            n_current = env._count_segment_vehicles("seg6") #number of vehicles in the merging area

            flows = env.get_last_measured_flows_vph()
            q_freeway = flows["q_freeway_vph"]
            q_ramp = flows["q_ramp_vph"]
            q_total_current = q_freeway + q_ramp

            if self.prev_n is None: 
                n = n_current 
                q_total = q_total_current
            else: 
                n = self.prev_n
                q_total = self.prev_q_total

            
            q_desire = q_total + self.Cf * (self.n_star - n)

            denom = q_desire - q_freeway 
            if abs(denom) < 1e-6:
                denom = 1e-6 
        
            
            c = (self.N / denom) * 3600.0

            self.cycle_time = max(1, int(round(c)))
            self.red_time = max(0, self.cycle_time - 1) 

            self.prev_n = n_current
            self.prev_q_total = q_total_current
        

        # ---- signal logic ----
        pos = self.t % self.cycle_time

        if pos < self.red_time:
            action = 0  # red
        else:
            action = 1  # green

        self.timer += 1
        if self.timer >= self.cycle_time:
            self.timer = 0

        self.t += 1
        return action
