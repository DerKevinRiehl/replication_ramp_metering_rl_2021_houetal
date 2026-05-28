# Replication study of "A Cyber-Physical System for Freeway Ramp Meter Signal Control Using Deep Reinforcement Learning in a Connected Environment"

## Authors
Julian Isser, Krishna Kanth Vuppala Narasimha, Kevin Riehl, Anastasios Kouvelas, Michail A. Makridis

Institute for Transportation Systems and Planning (IVT), Traffic Engineering Group, ETH Zürich.

This work is part of the ReScience C replication project conducted at IVT.

## Introduction
Freeway bottlenecks at on-ramp merging areas are one of the main sources of recurrent congestion in urban road networks. Ramp metering is the classical countermeasure: a traffic signal at the end of the on-ramp regulates how many vehicles are admitted to the mainline, which keeps the merging area below its capacity limit and the freeway from breaking down. Over the past decades, a variety of control strategies have been developed for this task, from fixed-time cycles and feedback controllers like ALINEA to learning-based approaches.

The reference study by Hou et al. (2021) applies three deep reinforcement learning algorithms (Ape-X DQN, PPO and A3C) to ramp metering control on a two-lane freeway with a single on-ramp and compares them against three baseline controllers (no ramp metering, fixed-time, ALINEA). The authors report that the RL controllers clearly outperform the baselines across six mobility and emission metrics, both in mixed near-congested/congested and in extremely congested traffic conditions. Since no source code was released alongside the paper, we re-implemented the entire pipeline from scratch (the network, the route generation, the environment, the baseline controllers, the three RL algorithms and the evaluation scripts) and used it to test whether the reported improvements can be reproduced.

## The replicated study.
```
Hou, Y., Zhang, X., Graf, P., Tripp, C., & Biagioni, D. (2021). A Cyber-Physical System for Freeway
Ramp Meter Signal Control Using Deep Reinforcement Learning in a Connected Environment.
In 2021 IEEE Intelligent Transportation Systems Conference (ITSC) (pp. 1–8). IEEE.
https://doi.org/10.1109/ITSC48978.2021.9564699
```

## What this repository includes
The repository follows the template structure of the replication project. The actual replication code lives in `1_code_produce/` and is organized by responsibility (environment, controllers, RL algorithms, evaluation scripts). The produced result files and the figures are kept under `2_data_produced/` and `3_data_visualization/` respectively, so the output of the experiments can be inspected without having to re-run them.

```
./
├── Readme.md
├── requirements.txt
├── template_info.md
├── 0_original_papers/                  Original Hou et al. (2021) paper
├── 0_original_repository/              Empty — the original authors did not publish code
├── 1_code_produce/                     All replication code
│   ├── network/ramp.net.xml            SUMO network (two-lane freeway, single on-ramp, seg1..seg10)
│   ├── config/config.sumocfg           Base SUMO configuration
│   ├── routes/
│   │   ├── vehicle_types.xml           Vehicle type definition (τ tuned to 1.04 s)
│   │   └── generate_routes.py          Mixed / extreme inflow sampling and route file writer
│   ├── environment/
│   │   ├── env.py                      Gymnasium environment wrapping SUMO via TraCI
│   │   └── space_speed_logger.py       Per-step spatial speed binning for Figure 6
│   ├── controllers/
│   │   ├── no_meter.py                 Always-green baseline
│   │   ├── fixed_time.py               3 s red / 1 s green cycle
│   │   └── alinea.py                   Feedback controller, re-evaluated every 15 s
│   ├── RLAlgorithms/
│   │   ├── PPO.py, APEXDQN.py, A3C.py  Training scripts (RLlib)
│   │   ├── train_rl.py                 Wrapper that launches a chosen training script
│   │   ├── evaluate_rl.py              Evaluation of trained PPO / Ape-X DQN policies
│   │   ├── evaluate_a3c.py             Separate evaluation script for A3C
│   │   ├── callbacks.py, early_stopping.py, function_utils.py, live_plot.py
│   │   └── ppo_runs/, apexdqn_runs/, a3c_runs/   Checkpoints and live reward plots
│   └── scripts/
│       ├── run_baselines.py                       Single-process baseline runner
│       ├── run_one_baseline_episode.py            One episode for one controller (used by the shell script)
│       ├── run_all_baselines_parallel.sh          Runs all 20 episodes × {noRM, Fixed, Alinea} in parallel
│       ├── merge_baseline_results.py              Merges per-episode CSVs into a single summary
│       ├── plot_paper_figures.py                  Reproduces Figures 4 and 5 (boxplots)
│       ├── plot_figure6.py                        Reproduces Figure 6 (spatial speed heatmap)
│       └── rl_means.py                            RL-vs-baseline summary numbers (Tables 4 and 6)
├── 1_data_source/                      Empty — all input data is generated in code (see generate_routes.py)
├── 2_data_produced/                    Per-episode CSVs, merged result tables, .npz spatial speed files
├── 3_code_visualization/               Plotting and aggregation scripts
└── 3_data_visualization/               Reproduced figures (Figure 4, 5, 6) and training reward curves
```

## Installation Instructions
The replication is written in Python and uses SUMO together with TraCI for the microscopic traffic simulation, RLlib for the reinforcement learning algorithms and Gymnasium for the environment interface. We tested the code with Python 3.11 and SUMO 1.26.

SUMO has to be installed separately and reachable via the `sumo` (or `sumo-gui`) command on the system path. We installed it manually from the official Eclipse SUMO download page:

```
# install SUMO from https://eclipse.dev/sumo/ (macOS / Windows / Linux installers)
# after the install, verify that `sumo --version` works in your shell

# clone the repository (this branch only) and install Python dependencies
git clone -b 2021_Hou_et_al https://github.com/DerKevinRiehl/eth_replication_project_rl_rm_work.git
cd eth_replication_project_rl_rm_work/4_template_project
pip install -r requirements.txt
```

The most relevant Python packages installed via `requirements.txt` are `ray[rllib]`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `torch`, and the SUMO bindings `sumo` and `traci`.

## Run Instructions
All commands below are executed from inside the `1_code_produce/` directory. The pipeline is split into four steps: running the baselines, training the RL agents, evaluating the trained policies and producing the figures.

### 1. Run the baseline controllers
The shell script runs all 20 evaluation episodes for the three baseline controllers in parallel and then merges the per-episode CSVs into a single summary table. Mixed and extreme traffic conditions are selected via the `--mode` argument inside the shell script.
```
bash scripts/run_all_baselines_parallel.sh
python -m scripts.merge_baseline_results
```
Output: `results/per_episode/result_*.csv`, `results/merged/baseline_results_all.csv`, `results/merged/baseline_results_summary.csv` and one `.npz` per episode in `results/space_speed/`.

### 2. Train the RL agents
Each RL algorithm is trained in its own process via the `train_rl.py` wrapper. The PPO and Ape-X DQN runs target roughly 20–40 M environment steps, A3C requires considerably more (we used up to 600 M).
```
python -m RLAlgorithms.train_rl --algo ppo
python -m RLAlgorithms.train_rl --algo apexdqn
python -m RLAlgorithms.train_rl --algo a3c
```
On 32 CPUs, PPO takes around 20 wall-clock hours to reach 40 M steps, Ape-XDQN about 7 hours for 40 M steps and A3C about 3 days for 600 M steps. Checkpoints are saved into `RLAlgorithms/{ppo,apexdqn,a3c}_runs/`, the best one is symlinked as `best_checkpoint/`. A live reward plot is written to the same directory.

### 3. Evaluate the trained policies
For PPO and Ape-X DQN:
```
python -m RLAlgorithms.evaluate_rl --algo ppo     --checkpoint RLAlgorithms/ppo_runs/last_checkpoint     --episodes 20 --traffic mixed
python -m RLAlgorithms.evaluate_rl --algo apexdqn --checkpoint RLAlgorithms/apexdqn_runs/last_checkpoint --episodes 20 --traffic mixed
```
A3C requires a separate evaluation script because the asynchronous workers cause issues when restoring the full algorithm object from a checkpoint:
```
python -m RLAlgorithms.evaluate_a3c --checkpoint RLAlgorithms/a3c_runs/last_checkpoint --episodes 20 --traffic mixed
```
Replace `--traffic mixed` with `--traffic extreme` to evaluate in the extreme scenario. Results are written to `results/per_episode/{algo}_{mode}.csv` and the spatial speed `.npz` files are written to `results/space_speed/{mode}/`.

### 4. Produce the figures and summary tables
```
python -m scripts.plot_paper_figures   # Figure 4 (mixed) and Figure 5 (extreme)
python -m scripts.plot_figure6         # Figure 6, spatial speed heatmap
python -m scripts.rl_means             # Numbers behind Tables 4 and 6
```
Figures are saved into `figures/` as both PNG and PDF.

## Replication Notes
A few points that are either not specified in the paper or required additional choices on our side are listed below.

- **Network.** The two-lane freeway with eight mainline segments and one on-ramp (seg9, seg10) was modelled in SUMO with the segment lengths given in Figure 1 of the paper. The merging area (seg6) is extended to roughly 150 m as in the paper; segment lengths are slightly reduced where a segment ends in a junction, because SUMO automatically absorbs part of the edge into the junction area.
- **Episode setup.** Each evaluation consists of 20 episodes of 1000 simulation steps with an additional 40-step warm-up phase during which no metrics are collected. Per-episode seeds are derived from the episode index so that all controllers see the same traffic realisations.
- **Inflows.** The mixed scenario draws the freeway flow from `N(3000, 300)` and the on-ramp flow from `N(900, 300)`, clipped to `[0, 3900]` and `[0, 1500]` vph respectively. The extreme scenario fixes the freeway flow at 3900 vph. Inflows are updated every 100 simulation steps. Note that SUMO does not accept a flow value of zero, so flows of zero are skipped in the route file rather than written out — otherwise the simulation terminates immediately.
- **Baselines.** No-RM keeps the signal green throughout the episode; the fixed-time controller uses the 3 s red / 1 s green cycle from the paper; ALINEA recomputes its cycle every 15 s based on Equations (1) and (2) with the paper's parameters (N=1, n*=10, C_f=20).
- **Tuned setup.** With the parameters exactly as given in the paper, the baseline metrics did not match the reported numbers. After empirical tuning we adjusted the vehicle headway to τ = 1.04 s, extended the fixed-time cycle to 15 s red / 1 s green, and averaged the ALINEA freeway and ramp inflows over a sliding window of 10 simulation steps. The RL agents were retrained on the tuned vehicle type but otherwise left unchanged.
- **RL agents.** All three algorithms use the same observation space (40 dimensions: mean speed and vehicle count per lane and segment, the previous signal phase, and the bottleneck throughput over the last 30 s), the same binary action space and the same reward function from Section IV-C of the paper. The neural network is a single hidden layer with 128 units and `tanh` activation. Hyperparameters (sample size, learning rate, exploration schedule, replay buffer size) follow Table I of the paper.
- **Training infrastructure.** Training runs use 32 parallel rollout workers on a single machine, as stated in the paper. The exact hardware Hou et al. used is not disclosed, which explains some of the difference in wall-clock time.
- **Fixed-time green phase.** The paper notes that only one car passes the merging area per green phase. SUMO does not allow the signal state to be modified within `traci.simulationStep()`, so this cannot be enforced explicitly — with the 3 s red / 1 s green cycle, however, the SUMO-GUI confirms that no more than one car passes per green phase in practice.
- **Spatial speed plots.** For Figure 6 the mainline (seg1–seg8) is discretised into 5 m bins. The environment optionally logs per-step vehicle positions and speeds into one `.npz` file per episode, and the plotting script aggregates them across episodes for each controller.

## Citation
Replication Study:
```
Isser, J., Vuppala Narasimha, K. K., Riehl, K., Kouvelas, A., & Makridis, M. A. (2026).
[RE] A Cyber-Physical System for Freeway Ramp Meter Signal Control Using Deep Reinforcement
Learning in a Connected Environment. ReScience C, 4(1).
```

Original Paper:
```
Hou, Y., Zhang, X., Graf, P., Tripp, C., & Biagioni, D. (2021). A Cyber-Physical System for Freeway
Ramp Meter Signal Control Using Deep Reinforcement Learning in a Connected Environment.
In 2021 IEEE Intelligent Transportation Systems Conference (ITSC) (pp. 1–8). IEEE.
https://doi.org/10.1109/ITSC48978.2021.9564699
```
