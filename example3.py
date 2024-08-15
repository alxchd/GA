import GA_binar as ga

# given problem parameters
parameters = {"rho": 7859*10**-9, #kg/mm**9
             "a" : 2000,  #mm
             "t" : 20,  #mm
             "h" : 40,  #mm
             "P" : 1000, #N
             "sigmaA" : 150, #MPa
             "E" : 2.1*10**5} #MPa

pop_size = 200

# run once, then run again on an shrunk domain, represented by a square with
# its width equal to 10% of the initial domain
n = 5
selection_types = ["boltzmann", "ruleta", "trunchiere", "tournament", "rank_based", "sus"]
# for each selection:
for index,selection in enumerate(selection_types):
    print(f'\nSelection type: {selection}\n')
    solutions = []
    for _ in range(n):
        population = ga.create_population(pop_size)
        temp_sol1 = ga.genetic_algorithm(pop_size, parameters, population,"sus")
        temp_sol1 = [ga.bin_to_dec(x) for x in temp_sol1]
        solution_fitness_1 = ga.objective_function(temp_sol1, parameters)
        print("Temporary solution is: ", temp_sol1, "with a fitness of: ", solution_fitness_1)
        solution_fitness_2 = solution_fitness_1 + 1
        infinite_loop_counter = 0
        while solution_fitness_1 < solution_fitness_2 and infinite_loop_counter < 10:
            population = ga.pop_radius(200, temp_sol1, pop_size)
            temp_sol2 = ga.genetic_algorithm(pop_size, parameters, population, "sus", temp_sol1)
            temp_sol2 = [ga.bin_to_dec(x) for x in temp_sol2]
            solution_fitness_2 = ga.objective_function(temp_sol2, parameters)
            infinite_loop_counter += 1
            
        solutions.append(temp_sol2)
        print("Final solution is: ", temp_sol2, "with a fitness of: ", solution_fitness_2)  