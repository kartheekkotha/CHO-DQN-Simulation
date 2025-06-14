# CHO-DQN-SIMULATION
[![DOI](https://zenodo.org/badge/1001927217.svg)](https://doi.org/10.5281/zenodo.15662973)


## Abstract
The impact of Radio link failure (RLF) has been largely ignored in designing handover algorithms, although RLF is a major contributor towards causing handover failure (HF). RLF can cause HF if it is detected during an ongoing handover. The objective of this work is to propose an efficient power control algorithm based on Deep-Q-learning (DQL), considering handover parameters (i.e., time-to-preparation, time-to-execute, preparation offset, execution offset) and radio link monitoring parameters (T310, T311, N310 and N311) as input. The proposed DRL based power control algorithm decides on a possible increase of transmitting power to avoid RLF driven HF. Simulation results show that the traditional conditional handover, when equipped with the proposed DRL based power control algorithm can significantly reduce both RLF and subsequent HF.   


## Features
CHO-DQN-SIMULATION is a Python-based simulation framework for evaluating Conditional Handover (CHO) in 5G NR networks, with a focus on modeling Radio Link Failures (RLF) and Handover Failures (HOF). It integrates a Deep Reinforcement Learning (DRL) agent to perform adaptive power control at the base stations (BS) to minimize RLF-driven HOF.

- Simulates multi-cell 5G NR network with configurable gNB deployments  
- Models UE mobility, log-distance path loss, Rayleigh fading, and dynamic obstacles  
- Implements 3GPP-compliant CHO and RLF detection (N310/T310 timers)  
- DRL agent (Double DQN) for power control decisions  
- Automatic logging of RLF, HOF, and power adjustments  
- Plotting utilities for reward curves, RLF/HOF count comparisons  


## Repository Structure

.
├── .gitignore  
├── Media/  
│   └── heatmap.png  
├── README.md  
├── requirements.txt  
└── src/  
    ├── config/  
    │   ├── eNB_param.py      # gNB deployment definitions  
    │   └── param.py          # simulation & RL hyperparameters  
    ├── tests/  
    │   ├── eNB_test.py       # eNB unit tests  
    │   ├── simulateUE_test.py  
    │   └── UE_test.py        # UE unit tests  
    ├── utils/  
    │   ├── grapher.py        # RSRP heatmap plotting  
    │   ├── misc.py           # conversion & calc helpers  
    │   └── Ticker.py         # simulation clock  
    ├── dqn_simulation.py     # DQN agent & replay buffer  
    ├── eNB.py                # base station model  
    ├── Simulate_UE.py        # simulation environment  
    ├── UE.py                 # UE mobility model  
    ├── test_agent.py         # run & compare agent vs baseline  
    ├── train_model.py        # agent training script  
    ├── overall_test.py       # full parameter-sweep runner  
    ├── plot_results_helper.py# bar-chart comparison utility  
    └── main.py               # (reserved / entry point)

## Installation

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `src/config/param.py` to tune:

- **Simulation**: `TOTAL_DISTANCE`, `MIN_SPEED`, `MAX_SPEED`, fading/obstacles flags  
- **CHO/RLF**: `Oprep`, `Oexec`, `Tprep`, `Texec`, `RLF_THRESHOLD`, `N310_THRESHOLD`, `T310`  
- **RL Agent**: learning rate, buffer size, ε-decay, agent indices & paths  

## Usage

1. **Train the DQN Agent**  
   ```bash
   python src/train_model.py
   ```
   - Trains over multiple episodes & parameter combinations  
   - Saves model in `train_logs/` and episodic reward CSVs  

2. **Evaluate Agent vs Baseline**  
   ```bash
   python src/test_agent.py
   ```
   - Runs before/after/default tests across chosen parameter sweep  
   - Outputs metrics & time-series plots in `test_logs/`

3. **Full Parameter Sweep**  
   ```bash
   python src/overall_test.py
   ```
   - Sweeps one parameter at a time (e.g. `Oprep`, `Texec`, `N310_THRESHOLD`)  
   - Saves per-run results & aggregate plots under `test_logs_overall/`

4. **Generate Comparison Bar Charts**  
   ```bash
   python src/plot_results_helper.py
   ```
   - Reads CSVs in a parent folder (e.g. `test_logs_overall/...`)  
   - Produces bar plots in `comparison_plots/`

5. **Utilities**  
   - `src/utils/grapher.py`: visualize RSRP heatmap  
   - `src/Simulate_UE.py`: environment entrypoint for custom scripts  

## Results

- Time-series plots, CSV logs, and comparison charts will be stored and are organized under:
  - `results_simulation/`  
  - `test_logs/`  
  - `test_logs_overall/`  
  - `train_logs/`

