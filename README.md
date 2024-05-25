# MPC-2.0

This repository is created for the assignment: A MPC Approach in Quadrotor. Below is the explanation of the files.

## **Src**
Source code file.
* LQR_sim.py: Simulate and visualize the result of input-constrained LQR for hovering at a fixed altitude.
* MPC_sim.py: Simulate and visualize the result of full-state feedback MPC for hovering at a fixed altitude.
### **Model**
* drone_dynamics.py:
  * Nonlinear drone dynamic
  * Linearization around the hovering equilibrium.
  * Full state feedback input constrained LQR definition.
  * Full state feedback MPC definition.
* test.py: Debug and testing file.
### **Planner**
* SafeFlightPolytope.py: Generate state constraints based on the environment.
* trajectory_generation.py: Generate the reference trajectory for simulations.

## **PDF**

## **Text**
*TheoryFootnote*: Note from reading the MPC textbook about proving stability.
