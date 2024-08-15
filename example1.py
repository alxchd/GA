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

# run the algorithm n times for each type of selection indicated in the list
# choose one or more: ["roulette", "truncation", "tournament", "rank_based", "sus", "boltzmann"]
selection_types = ["roulette"]
n = 10
#run
for index, selection in enumerate(selection_types):
    gen_z = []
    for _ in range(n):
        population = ga.create_population(pop_size)
        b, c = ga.genetic_algorithm(pop_size, parameters, population, selection)
        b,c=ga.bin_to_dec(b),ga.bin_to_dec(c)
        gen_z.append((b, c))
print("Has converged to the global optimum:")
print((17,1200) in gen_z)