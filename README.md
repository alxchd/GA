# GA
Genetic Algorithm for Two-Variable Optimization Problems

This repository contains an implementation of a genetic algorithm (GA) designed to solve two-variable optimization problems. The algorithm allows for experimentation by modifying parameters like the number of generations, mutation rates, and selection methods. The code is organized into four primary files: GA_binary.py, example1.py, example2.py, and example3.py.

Repository Structure
1. GA_binary.py:
   - Purpose: This file contains the core genetic algorithm and utility functions required to execute it.
   - Key Function: 
     - genetic_algorithm(pop_size, parameters, population, selection, solution=None): The main subroutine that drives the algorithm.
       - pop_size: Number of individuals in each generation.
       - parameters: Dictionary of problem-specific constants.
       - population: Initial list of individuals.
       - selection: Selection strategy for choosing parents. Available options include ["roulette", "truncation", "tournament", "rank_based", "sus", "boltzmann"].
       - solution: An optional known suboptimal solution to help the algorithm converge more effectively.
   - Usage:
     - Start by generating the initial population using the create_population(pop_size) function.
     - Modify parameters directly in the code, such as the number of generations (G), mutation and selection functions to observe their impact on the solutions precision.

2. example1.py:
   - Purpose: Demonstrates the GAs stochastic nature by running the algorithm multiple times (n runs) to increase the likelihood of finding the global optimum.
   - Usage: Use this script to establish baseline performance and gather results that can serve as references for further experimentation.

3. example2.py:
   - Purpose: Enhances the search by running the algorithm n-1 times, then using the intermediate solutions to form a final population for one last run.
   - Usage: This approach may lead to better convergence by refining the population before a final optimization run.

4. example3.py:
   - Purpose: This script narrows the search space by first performing an initial run, then shrinking the search domain based on the solution found, and finally conducting a more focused search.
   - Usage: Particularly useful for problems where a rough estimate of the solution can significantly reduce the search space, leading to faster convergence.

Important Considerations
- Generations (G): The number of generations is a critical parameter that influences how well the algorithm converges. Experiment with this to balance between computational time and solution precision.
- Mutation and Selection: These functions are hardcoded in the current implementation. To experiment with different strategies, youll need to modify the source code.
- Problem Adaptability: While the current implementation is tailored for a two-variable optimization problem, adapting it to other types of problems will require substantial modifications.

Example Problem
The provided example problem is optimized to converge to the solution package (17, 1200). You can modify the parameters and run the examples to see how different strategies impact the convergence and solution quality.

Quality of Solutions
The primary metric for evaluating solution quality is the incidence of the algorithm converging to the global optimum. Since genetic algorithms are probabilistic, running the algorithm multiple times can provide a better chance of finding the best possible solution.

---

Feel free to explore and modify the code to better suit your optimization problem. This readme provides a starting point, but the flexibility of the GA allows for extensive customization based on your needs.
