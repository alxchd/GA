import random as rnd
from math import sqrt,exp,pi
import numpy as np


def dec_to_bin(decimal):
    """
    Takes in a integer decimal and outputs its 11 bit binary version. 

    """ 
    n = 11
    string = np.binary_repr(decimal)
    binary = np.array([])
    for digit in string:
        binary = np.append(binary,int(digit))
    # pad binary to length 11
    length = len(binary)
    if length < n:
        pad = np.zeros((1, n - length))
        binary = np.append(pad,binary)
    return binary

def objective_function(individual ,parameters):
    """
    Calculates the fitness of the current individual.
    """ 
    b = individual[0]
    c = individual[1]
    l23=sqrt(c**2+b**2)
    return parameters["rho"]*parameters["t"]*parameters["h"]*(parameters["a"]+l23)

def constraints(individual,parameters):
    """
    Cheks if the individual meets the set constraints.
    """ 
    b = individual[0]
    c = individual[1]
    Mmax = parameters["P"]*(parameters["a"]-c)
    l23=sqrt(c**2+b**2)
    N23 = parameters["P"]*parameters["a"]/c/(b/l23)
    Iz = parameters["t"] * parameters["h"]**3 / 12
    Pc = pi**2*parameters["E"]*Iz/l23**2
    return 6*Mmax/(parameters["t"]*parameters["h"]**2*parameters["sigmaA"])<=1    \
        and N23<=parameters["sigmaA"]*parameters["t"]*parameters["h"] and N23 < Pc
        
def roulette_selection(dec_population, pop_size, parameters):
    """
    Performs roullette-wheel selection.
    """ 
    # check the feasibility of the population
    feasible_individuals = [individual for individual in dec_population \
                         if constraints(individual, parameters)]

    # fitness index for roulette bias
    fitness_bias = [objective_function(individual, parameters) for individual in feasible_individuals]
    
    # extract k parents from feasible individuals, biasing according to fitness_bias
    dec_parents = rnd.choices(feasible_individuals, weights=fitness_bias, k=pop_size)
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in dec_parents]
    
    return parents

def truncation_selection(dec_population, pop_size, parameters):
    """
    Performs truncation selection.
    """ 
    # k is the number of individuals to be selected
    k = 30
    # check the feasibility of the population
    feasible_individuals = [individual for individual in dec_population \
                          if constraints(individual, parameters)]
   
    # choose top k parents only
    dec_parents = sorted(feasible_individuals, key = (lambda individual: objective_function(individual, parameters)))
    dec_parents = dec_parents[:k-1]
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in dec_parents]
    return parents

def tournament_selection(dec_population, pop_size, parameters, kh=30):
    """
    Performs tournament selection.
    kh: Type int - cohort size
    """ 
    parents = []
    
    # perform selection on feasible individuals only
    feasible_individuals = [individual for individual in dec_population \
                          if constraints(individual, parameters)]
   
    while len(parents) < pop_size:
        # form tournament cohorts by picking kh individuals from the feasible candidates and
            #add the fittest to the parents collector
        parents_potentiali = rnd.choices(feasible_individuals, k = kh)
        parents.append(sorted(parents_potentiali, key = (lambda individual: objective_function(individual, parameters)))[0])
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in parents]
    return parents

def rank_based_selection(dec_population, pop_size, parameters):
    """
    Performs rank-based selection.
    """ 
    # perform selection on feasible individuals only
    feasible_individuals = [individual for individual in dec_population \
                         if constraints(individual, parameters)]
    # sort and rank for bias based on rank
    feasible_individuals.sort(reverse=True, key = (lambda individual: objective_function(individual, parameters)))
    ranks = list(range(1,len(feasible_individuals)+1))
    dec_parents = rnd.choices(feasible_individuals, weights=ranks, k=pop_size)
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in dec_parents]
    return parents

def SUS_selection(dec_population, pop_size, parameters):
    """
    Performs Stochastic Universal Sampling.
    """ 
    parents = []
    # perform selection on feasible individuals only
    feasible_individuals = np.array([individual for individual in dec_population \
                         if constraints(individual, parameters)])
    # index de fitness pentru calculul pointerului
    fitness_bias = np.array([objective_function(individual, parameters) for individual in feasible_individuals])
    # pastreaza elita
    # index_elita = np.argmin(fitness_bias)
    # sorteaza indivizii fezabili impreuna cu indicii lor
    indici_sort = np.argsort(fitness_bias)
    feasible_individuals, fitness_bias = feasible_individuals[indici_sort],100/fitness_bias[indici_sort]
    # aseaza pe axa fitnessul
    cumfit = np.cumsum(fitness_bias)
    # genereaza pointers
    start = rnd.uniform(0,fitness_bias[0])
    pasul = (cumfit[-1]-fitness_bias[0])/(pop_size+1)
    # alege indivizii
    poz = start
    for _ in range(pop_size):
        indexales = np.searchsorted(cumfit, poz)
        parents.append(feasible_individuals[indexales])
        poz+=pasul
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in parents]
    return parents

def boltzmann_selection(dec_population, pop_size, parameters):
    """
    Performs Boltzmann selection.
    """ 
    parents = []
    # perform selection on feasible individuals only
    feasible_individuals = np.array([individual for individual in dec_population \
                         if constraints(individual, parameters)])
    # fitness index for pointer calculation
    fitness_bias = np.array([objective_function(individual, parameters) for individual in feasible_individuals])
    # calculate selection probability
    k = 1 + 100*generation/G
    alfa = 0.5
    T0 = 6
    T = T0*(1-alfa)**k
    while len(parents) < pop_size:
        for individual in feasible_individuals:
            if objective_function(individual,parameters) < fitness_history[-1]:
                parents.append(individual)
            else:
                P = exp(-(objective_function(individual,parameters) - min(fitness_bias))/T)
                # print(P) # debugging
                if rnd.random() < P:
                    parents.append(individual)
                    # print("taken") # debugging
    parents = [(dec_to_bin(parent[0]),
                dec_to_bin(parent[1])) for parent in parents]
    return parents

def crossover(parents, pop_size):
    """
    Performs crossover.
    
    Parameters
    ----------
    parents : array
    pop_size : int

    Returns
    -------
    offspring : array

    """
    offspring = []
    zero = np.zeros(11)
    while len(offspring) < pop_size:
        child = [1,2]
        for i in range(pop_size):
            # pick two random parents
            parent1, parent2 = rnd.choices(parents, k=2)
            # initialise a child with the values of parent1
            child = parent1
            # randomly replace genes in child with genes from parent2
            for i in range(len(child)):
                if rnd.random() < 0.5:
                    child[0][i]=parent2[0][i]
                if rnd.random() < 0.5:
                    child[1][i]=parent2[1][i]
            # verifica daca child are vreun zero si reia daca e adevarat
            # check if a genes equivalent to decimal 0 have resulted and repeat the process if so
            if np.array_equal(child[0], zero) or np.array_equal(child[1], zero):
                # print('<debugging> zero value created in crossover')
                continue
            else:
                # print(child)
                offspring.append(child)
    return offspring

def mutation(generation, offspring, pop_size):
    """
    Performs mutation

    Parameters
    ----------
    generation : int
        The current generation.
    offspring : array
        The collector of the new population (the next generation).
    pop_size : int
        The size of the populations.

    Returns
    -------
    offspring : array
        The collector of the new, mutated population (the next generation).
    """
    # calculate mutation probablility
    mutation_probability = 1/generation*10
    # prepare to check for values equivalent to decimal zero
    zero = np.zeros(11)
    # iterate gene by gene and mutate where appliable
    for index1, individual in enumerate(offspring):
        for index2, gena in enumerate(individual):
            while True:
                for i in range(len(gena)):
                    if rnd.random() < mutation_probability:
                        if gena[i] == 1:
                            individual[index2][i] = 0
                        else:
                            individual[index2][i] = 1
                if np.array_equal(gena, zero) == False:
                    break
            # append non zero characteristic to individual
        offspring[index1] = individual
    return offspring

def bin_to_dec(binar):
    """
    Converts a binary array into a decimal number

    Parameters
    ----------
    binar : array

    Returns
    -------
    int(decimal): int

    """
# convert a binary array to decimal
    decimal = 0
    power = len(binar)-1
    for digit in binar:
        decimal += digit * 2 ** power
        power += -1
    return int(decimal)

def create_population(pop_size):
    """
    Creates the starging GA population. The function will only work correctly
    in the current context.
    b and c are parameters of the mathematical model of the problem.

    Parameters
    ----------
    pop_size : int
        It represents the nuber of individuals in the population.

    Returns
    -------
    population : array
        An array consisting of individuals and their genotype.

    """
    population = []
    i=0
    while i < pop_size:
        # create random, binary, unidimensional arrays, of size 11
        b = np.array([])
        for j in range(11):
            b = np.append(b, rnd.randint(0, 1))
        c = np.array([])
        for j in range(11):
            c = np.append(c, rnd.randint(0, 1))
        # append only if the values belong to the domain of the problem
        if (bin_to_dec(b) < 2000
                and bin_to_dec(c) < 2000 and bin_to_dec(b)!=0 and bin_to_dec(c) !=0):
            population.append((b,c))
            # raise the meter if appended
            i+=1
    return population
    
def genetic_algorithm(pop_size, parameters, population, selection, solution=None):
    """
    Solves a given genetic optimization problem. Variabile G - generations to
    be simulated - can be modified to observe its influience on the solution
    precision. The current problem has the following solution package: (17,1200)

    Parameters
    ----------
    pop_size : int
        The number of individuals in the populations.
    parameters : dict()
        Constants characteristic to the current problem to be solved.
    population : list
        A list containing all the individuals in the population.
    selection : string
        One of the following options are available:
            ["roulette", "truncation", "tournament", "rank_based", "sus", "boltzmann"]
    solution : touple, optional
        Provide a pair of values if a suboptimal solution is known. The default is None.

    Returns
    -------
    touple pair of ints
        Will convert the binars within the solution to decimals and will return them.

    """
    
    # Initialization of the algorithm
    global generation # will be necessary for mutation and boltzmann_selection
    global G # the total number of generations to iterate
    generation = 0
    solution_fitness = None#float('-inf') #  
    # in var ce urmeaza vor fi stocate solutiile per generatii. E necesar a fi initializat pentru
    # a introduce iteratia in algoritm
    global fitness_history # necesar a fi global in selection Boltzmann
    fitness_history = [float('inf')]
    solution_history = []
    if solution is not None:
        solution_history.append(solution)
        fitness_history.append(objective_function(solution, parameters))
    
    # convert the population to decimal
    dec_population = [(bin_to_dec(individual[0]), bin_to_dec(individual[1])) for individual in population]
    
    # total generations to be run
    G = 100
    
    while len(fitness_history)<G+1 and generation < G: 
        generation+=1
            
        # selection
        if selection == "roulette":
            parents = roulette_selection(dec_population, pop_size, parameters)
        elif selection == "truncation":
            parents = truncation_selection(dec_population, pop_size, parameters)
        elif selection == "tournament":
            parents = tournament_selection(dec_population, pop_size, parameters)
        elif selection == "rank_based":
            parents = rank_based_selection(dec_population, pop_size, parameters)
        elif selection == "sus":
            parents = SUS_selection(dec_population, pop_size, parameters)
        elif selection == "boltzmann":
            parents = boltzmann_selection(dec_population, pop_size, parameters)
            
        # parents crossover
        offspring = crossover(parents, pop_size)
    
        # offspring mutation
        offspring = mutation(generation, offspring, pop_size)
        
        # urmatoarele operatii au loc in decimal, converteste offspring
        # convert offspring to decimal for the following operations
        offspring_dec = [(bin_to_dec(individual[0]), bin_to_dec(individual[1])) for individual in offspring]
        
        # elitism
        if solution is not None:
            # replace the least fit individual from offspring with the current solution
            offspring_dec[np.argmax([objective_function(individual, parameters) for individual in offspring_dec])] = solution
        # offspring becomes the current population
        dec_population = offspring_dec
    
        # try the solutions        
        feasible_solutions = [individual for individual in dec_population if constraints(individual,parameters)]
        # itereaza prin solutii fezabile, incearca-le cu functia obiectiv si returneaza solution care da cel mai bun rezultat
        # iterate through the feasible solutions and extract the fittest individual
        solution = min(feasible_solutions, key = lambda individual:objective_function(individual, parameters))
        solution_fitness = objective_function(solution,parameters)
        # if more fit than the best solution, append it
        if objective_function(solution, parameters) < fitness_history[-1]:
            solution_history.append(solution)
            fitness_history.append(solution_fitness)
            
    # return the solution in a decimal set of values
    return (dec_to_bin(solution_history[-1][0]), dec_to_bin(solution_history[-1][1]))

def pop_radius (diam, center, pop_size):
    pop = [] # populația
    radius = diam/2
    l_bounds = np.array(center) - radius
    for i,v in enumerate(l_bounds):
        if v < 0:
            l_bounds[i] = 1            
    u_bounds = np.array(center) + radius
    for i,v in enumerate(u_bounds):
        if v > 2000:
            u_bounds[i] = 2000
    while len(pop) < pop_size:
        x = rnd.randint(l_bounds[0], u_bounds[0])
        y = rnd.randint(l_bounds[1], u_bounds[1])
        pop.append((x,y))
    pop_binar = [(dec_to_bin(a), dec_to_bin(b)) for (a,b) in pop]
    return pop_binar