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

# run the algorithm on the gen_m population, consisting of m solutions
n=9
solutions = []
# choose one or more from the selection types:
    #["roulette", "truncation", "tournament", "rank_based", "sus", "boltzmann"]
selection_types = ["roulette"]
for index,selection in enumerate(selection_types):
    print(f'\nSelection type: {selection}\n')
    solutions = []
    gen_z = []
    # generate n solutions
    for _ in range(n):
        m = 5
        gen_m = []
        # one iteration 
        for _ in range(m):
            population = ga.create_population(pop_size)
            b, c = ga.genetic_algorithm(pop_size, parameters, population, selection)
            gen_m.append((b,c))
        # run the algorithm on gen_m
        final_solution = ga.genetic_algorithm(pop_size, parameters, gen_m, selection)
        gen_z.append(final_solution)
        gen_z_decimal = []
        # convert to decimal
        for solutie in gen_z:
            x = ga.bin_to_dec(solutie[0])
            y = ga.bin_to_dec(solutie[1])
            gen_z_decimal.append((x,y))
        # extract the improved solution
        solution = min(gen_z_decimal, key = lambda individual:ga.objective_function(individual, parameters))
        print(f'Identified solution: {solution}')
        solutions.append(f'{solution[0]}, {solution[1]}')