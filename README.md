budget-pcsf-semigradient-ascent
==================
Heuristic algorithms for solving the budget-constrained prize collecting Steiner forest problem.
scr-reserve-design

* **Citation Info:** Gupta, A. and Dilkina, B., 2019, November. Budget-Constrained Demand-Weighted Network Design for Resilient Infrastructure. In *2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)*. IEEE.

--------
Overview
--------
We address the problem of allocating a fixed budget towards restoring edges to maximize the satisfied travel demand between locations in a network, which we formalize as the budget-constrained prize-collecting Steiner forest problem. We prove that the satisfiable travel demand objective exhibits *restricted supermodularity over forests*, and utilize this property to design an iterative algorithm based on maximizing successive modular lower bounds for the objective that finds better solutions than a baseline greedy approach. We also propose an extremely fast heuristic for maximizing modular functions subject to knapsack and graph matroid constraints that can be used as a subroutine in the iterative algorithm, or as a standalone method that matches the greedy baseline in terms of quality but is hundreds of times faster. We evaluate the algorithms on synthetic data, and apply them to a real-world instance of retrofitting the Senegal national and regional road network against flooding.
