# import random as rnd
# import matplotlib.pyplot as plt
import GA_binary as ga
from math import sqrt
import numpy as np
import PostProcessing as pp



# given parameters
parameters = {"rho": 7859*10**-9, #kg/mm**9
             "a" : 2000,  #mm
             "t" : 20,  #mm
             "h" : 40,  #mm
             "P" : 1000, #N
             "sigmaA" : 150, #MPa
             "E" : 2.1*10**5} #MPa

pop_size = 500

# FEASIBILITY STUDY FOR THE DEFINED PROBLEM
    # will run a series of optimizations on a loads sweep, which it will write to 
    # a output.txt. Wil then plot the mass of the baseline vs the mass of the 
    # optimized concept as functions of load.
    
filename = "output.txt"
solutions = []
feasibility = []
loads = np.arange(500,2050,50)
selections = ["rank_based"]
for P in loads:
    parameters["P"] = P
    m_baselin = parameters["rho"]*parameters["t"]*sqrt(6*P*parameters["a"]/(parameters["t"]*parameters["sigmaA"]))*parameters["a"]

    for selection in selections:
        temp_solutions = []
        population = ga.create_population(pop_size)
        maybe_sol = ga.genetic_algorithm(pop_size, parameters, population,selection, None)
        maybe_sol = [ga.bin_to_dec(x) for x in maybe_sol]
        temp_solutions.append(maybe_sol)
    sorted(temp_solutions,key=lambda x: ga.objective_function(x, parameters))
    solutions.append(temp_solutions[-1])
    print("Found solution::", temp_solutions[-1])
    content = f"Function value: {ga.objective_function(temp_solutions[-1], parameters):.4f} [kg] vs {m_baselin:.4f} [kg]\n"
    pp.write_to_text_file(filename, content)
    feasibility.append(2-ga.objective_function(temp_solutions[-1], parameters)/m_baselin)    
    # plot
pp.plot_res_feas_study(filename,loads,check=False)