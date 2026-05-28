
class NoMeterController: 

    
    #Baseline Controller with no ramp metering 
    #Always allows vehicles to enter the freeway, as the traffic light is always green


    def __init__(self):
        pass 

    def compute_action(self, state=None):

        #no ramp meter policy: always green (0 for red, 1 for green in the action space)

        return 1
    
    def reset(self): #trivial reset 
        return 