import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self, save_path = None, algo_name="ppo"):
        plt.ion()  # interactive mode ON
        self.save_path = save_path
        self.algo_name = algo_name

        self.fig, self.ax = plt.subplots()

        self.x_data = []
        self.y_data = []

        # --- color & title depending on algorithm ---
        if self.algo_name == "ppo":
            color = "#1f77b4"  # blue
        elif self.algo_name == "a3c":
            color = "#d62728"  # red
        elif self.algo_name == "apexdqn":
            color = "#ff7f0e"  # orange
        else:
            color = "#1f77b4"  # default is blue 

        # colors like in paper 
        self.line, = self.ax.plot(
            [], [],
            color=color,
            linewidth=2,
            label="Reward"
        )

        self.ax.set_xlabel("Training steps")
        self.ax.set_ylabel("Reward")
        self.ax.set_title(f"{self.algo_name} Training")

        #grid like in paper 
        self.ax.grid(True, linestyle="--", alpha=0.5)

        self.ax.legend()

    def update(self, x, y):
        self.x_data.append(x)
        self.y_data.append(y)

        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.save_path is not None:
            self.fig.savefig(self.save_path, dpi=300, bbox_inches="tight")