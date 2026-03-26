# MPC-2.0

A Model Predictive Control (MPC) approach for quadrotor control. This project implements and compares input-constrained LQR, full-state feedback MPC, and observer-based output feedback MPC for quadrotor hovering and trajectory tracking.

## Setup

```bash
conda env create -f environment.yml
conda activate scientific
```

## Running Simulations

All scripts are run from the repository root:

```bash
python -m src.LQR_sim                              # Input-constrained LQR hovering
python -m src.MPC_sim                               # MPC hovering
python -m src.simulations.simulation                # MPC with optimal target selection
python -m src.simulations.linear_vs_nonlinear       # Linear vs nonlinear dynamics comparison
python -m src.simulations.MPC_outputFeedback_sim    # Observer-based output feedback MPC
python -m src.estimateXf                            # Compute maximal control invariant set
```

## Project Structure

### src/

- **LQR_sim.py** — Simulate and visualize input-constrained LQR for hovering at a fixed altitude.
- **MPC_sim.py** — Simulate and visualize full-state feedback MPC for hovering at a fixed altitude.
- **estimateXf.py** — Compute the maximal control invariant admissible set Xf for MPC terminal constraints.

### src/model/

- **drone_dynamics.py** — Core quadrotor model:
  - 12-state nonlinear 6-DOF dynamics (position, attitude, velocities, angular rates)
  - CasADi-based linearization around operating points
  - LQR, constrained LQR, and MPC controllers
  - Optimal target selector for steady-state reference computation
  - Luenberger observer design via Kalman filtering for state and disturbance estimation

### src/planner/

- **trajectory_generation.py** — Reference trajectory generators (hover, circular, square, waypoint-based, linear interpolation).

### src/simulations/

- **simulation.py** — MPC simulation with optimal target selection.
- **linear_vs_nonlinear.py** — Side-by-side comparison of linearized vs full nonlinear dynamics under MPC.
- **MPC_outputFeedback_sim.py** — Output feedback MPC with augmented disturbance estimation and Luenberger observer.

### Other

- **TheoryFootnote.txt** — Notes on MPC stability theory from textbook readings.
- **figure/** — Output visualization images.
