
class RewardPlateauEarlyStopping:

    #defines an early stopper for the RL training 

    def __init__(self, min_iterations=30, patience=5, min_delta=0.5, smoothing_window=5):
        self.min_iterations = min_iterations
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window

        self.reward_history = []
        self.best_smoothed_reward = None
        self.bad_iterations = 0

    def update(self, reward_mean):

        if reward_mean is None: #if argument is none then return immediately 
            return False

        self.reward_history.append(float(reward_mean))

        recent = self.reward_history[-self.smoothing_window:] #returns the last 5 rewards 
        smoothed = sum(recent) / len(recent)

        if len(self.reward_history) < self.min_iterations: #too early 
            return False

        if self.best_smoothed_reward is None:
            self.best_smoothed_reward = smoothed
            return False

        if smoothed - self.best_smoothed_reward > self.min_delta: #improvement was there 
            self.best_smoothed_reward = smoothed
            self.bad_iterations = 0
        else:
            self.bad_iterations += 1 #too little improvement 

        return self.bad_iterations >= self.patience #did we have enough many bad iterations? 

