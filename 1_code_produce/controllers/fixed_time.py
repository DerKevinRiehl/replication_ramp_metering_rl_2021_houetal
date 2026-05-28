import traci
class FixedTimeController: 

    
   #Fixed-Time Ramp Meter Controller with 3 seconds red (0) and 1 second green (1), then repeat


    def __init__(self):
        self.t = 0 #internal counter for timesteps 
    
    def reset(self): #reset internal timer at the start of each episode
        self.t = 0
        

    def compute_action(self, state=None):

        cycle_position = self.t % 4
        
        if cycle_position < 3: #timestep 0,1,2 --> red 

            action = 0

        else:
            action = 1 #timestep 3 --> green
        
        self.t += 1 

        return action 

    
    