# FSCM_TMS

Project for TMS Simulation of head models using the ficticious surface charge method. Within this project the Fast Multipole method is implemented to solve the maxwell  equations to find the field on any surface in a multilayer 3D surface mesh. For this the electric charge distribution is computed using the surface charge simulation method (1), during which the resulting matrix creation is optimized with efficient sparse matrix
assembly and the solution of the linear algebra problem with this matrix is optimized by using iterative solvers like the Jacobi Method or GMRES and executing the code on a GPU using the Cupy module.

This project was developed during a master's thesis.

The paper (2) was build entirely on this code, where the main results where computed in Head_model_TUI.py and Q_methods_Testbench.py, the latter was plotted in Plot_results_Q_methods.py 

Almost all functions of the project are contained in functions.py. The GPU versions of various functions are marked with _cu or _cupy as suffix. Documentation is sparse due to time constraints and lack of experience. A class based version of some functions has been implemented in TMS.py and can be tested from TMS_example_clean.py


(1):  B. Petkovic, K. Weise, and J. Haueisen, “Computation of Lorentz force
and 3-D eddy current distribution in translatory moving conductors in
the field of a permanent magnet,” IEEE Trans. Magn., vol. 53, no. 2,
pp. 1–9, Feb. 2017.

(2): Müller, E., Petković, B., Ziolkowski, M., Wise, K., Toepfer, H.,
& Haueisen, J. (2023). An Improved GPU Optimized Fictitious Surface
Charge Method for Transcranial Magnetic Stimulation. IEEE Transactions on Magnetics.