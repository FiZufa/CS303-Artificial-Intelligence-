import argparse
import networkx as nx
import random
import numpy as np

def read_social_network(file_path):
    social_graph = nx.DiGraph()
    
    with open(file_path, 'r') as file:
        n, m = map(int, file.readline().split())
        
        for _ in range(m):
            u, v, weight_p1, weight_p2 = map(float, file.readline().split())
            
            social_graph.add_edge(int(u), int(v), p1=weight_p1, p2=weight_p2)
    
    return n, m, social_graph

def read_seed_set(file_path):
    seed_set = {
        'k1': 0,
        'k2': 0,
        'c1_seeds': set(),
        'c2_seeds': set()
    }

    with open(file_path, 'r') as file:
        k1, k2 = map(int, file.readline().split())
        seed_set['k1'] = k1
        seed_set['k2'] = k2

        for i in range(k1):
            seed = int(file.readline())
            seed_set['c1_seeds'].add(seed)

        for i in range(k2):
            seed = int(file.readline())
            seed_set['c2_seeds'].add(seed)

    return seed_set


def formulate_diffusion(graph, seed_sets, edge_prob):
    node_active = {nd: False for nd in graph.nodes()}
    exposed_nodes = set()
    bfs_queue = list(seed_sets)

    for nd in seed_sets:
        node_active[nd] = True
        exposed_nodes.add(nd)

    while bfs_queue:
        current_node = bfs_queue.pop(0)
        # Check if the node exists in the graph before accessing its successors
        if current_node in graph:
            for outward_node in graph.successors(current_node):
                if node_active[outward_node] is False:
                    p = edge_prob.get((current_node, outward_node), 0.0)

                    if random.random() <= p:
                        node_active[outward_node] = True
                        bfs_queue.append(outward_node)

                    exposed_nodes.add(outward_node)

    activated_nodes = len([nd for nd in node_active.values() if nd])
    return activated_nodes, exposed_nodes



def objective_value(population, graph, initial_seed, nodes_length, budget, num_sim):

    balance_c1 = []
    balance_c2 = []

    for i in range(nodes_length):
        if population[i] == 1:
            balance_c1.append(i)

    
    
    for i in range(nodes_length):
        if population[nodes_length + i] == 1:
            balance_c2.append(i)

    combined_c1_seeds = initial_seed['c1_seeds'] | set(balance_c1)
    combined_c2_seeds = initial_seed['c2_seeds'] | set(balance_c2)

    sum = 0

    for i in range(num_sim):
        activated_c1, exposed_c1 = formulate_diffusion(graph, combined_c1_seeds, nx.get_edge_attributes(graph, 'p1'))
        activated_c2, exposed_c2 = formulate_diffusion(graph, combined_c2_seeds, nx.get_edge_attributes(graph, 'p2'))

        union_c1 = set(exposed_c1).union(set(balance_c1))
        union_c2 = set(exposed_c2).union(set(balance_c2))

        sym = set(union_c1 - union_c2).union(set(union_c2-union_c1))
        phi = len(set(graph.nodes).difference(set(sym)))

        sum += phi

    obj_value = sum / num_sim

    return obj_value


def calculate_fitness(population, graph, initial_seed, nodes_length, budget, num_sim):

    fitness_val = 0
    sigma_x = np.count_nonzero(population)
    #print('sigma_x: ' + str(sigma_x))

    if (sigma_x <= budget):
        fitness_val = objective_value(population, graph, initial_seed, nodes_length, budget, num_sim)
    
    else:
        fitness_val = -sigma_x
    
    return fitness_val


def roulette_wheel_selection(population, fitness_values):
    # Calculate probabilities
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness

    # Construct the roulette wheel
    wheel = np.cumsum(probabilities)

    # Select two parents
    selected_parents = []
    for _ in range(2):
        spin = random.random()
        selected_index = np.searchsorted(wheel, spin)
        #print('selected index: ' + str(selected_index))
        selected_parents.append(population[selected_index])

    return selected_parents


def one_point_crossover(parent1, parent2):
    
    crossover_point = random.randint(1, len(parent1) - 1)

    # Create offspring by swapping genetic material
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return offspring1, offspring2


def mutate(individual, mutation_probability):
    mutated_individual = individual.copy()

    for i in range(len(mutated_individual)):
        if random.random() < mutation_probability:
            # Mutate the gene (change its value)
            mutated_individual[i] = 1 - mutated_individual[i]

    return mutated_individual


def problem_formulation(nodes_length, population_size, graph, initial_seed, budget, num_sim):

    #activate = np.zeros(2*nodes_length, dtype=int)
    population = list()
    for i in range(population_size):
        balanced_set = np.zeros(2*nodes_length, dtype=int) 

        for j in range(budget):
            index = random.randint(0, 2 * population_size - 1)
            balanced_set[index] = 1

        population.append(balanced_set)


    fitness_values = np.zeros(population_size, dtype=int)

    for i in range(population_size):
        #def calculate_fitness(population, graph, initial_seed, nodes_length, budget, num_sim):
        fitness_values[i] = calculate_fitness(population[i], graph, initial_seed, nodes_length, budget, num_sim)

    generations = 5
    #print(fitness_values)
    offsprings_array = [] #for each iteration, get new 2 offsprings
    for gen in range(generations):
        selected_parents = roulette_wheel_selection(population, fitness_values)
        offspring1, offspring2 = one_point_crossover(selected_parents[0], selected_parents[1])
        mutation_probability = 0.01 
        mutated_offspring1 = mutate(offspring1, mutation_probability)
        mutated_offspring2 = mutate(offspring2, mutation_probability)
        offsprings_array.append(mutated_offspring1)
        offsprings_array.append(mutated_offspring2)

    new_solution = population + offsprings_array

    new_fitness = np.zeros(len(new_solution), dtype=int)

    for i in range(len(new_solution)):
        new_fitness[i] = calculate_fitness(new_solution[i], graph, initial_seed, nodes_length, budget, num_sim)
    
    # for i in range(len(new_solution)):
    #     print('new fitness: ' + str(new_fitness[i]) + ' ')
    new_offspring_array = []
    for gen in range(generations):
        selected_parents = roulette_wheel_selection(new_solution, new_fitness)
        offspring1, offspring2 = one_point_crossover(selected_parents[0], selected_parents[1])
        mutation_probability = 0.01 
        mutated_offspring1 = mutate(offspring1, mutation_probability)
        mutated_offspring2 = mutate(offspring2, mutation_probability)
        new_offspring_array.append(mutated_offspring1)
        new_offspring_array.append(mutated_offspring2)

    #best_parent = roulette_wheel_selection(new_solution)

    final_solution = new_offspring_array + new_solution
    final_fitness = np.zeros(len(final_solution), dtype=int)

    for i in range(len(final_solution)):
        final_fitness[i] = calculate_fitness(final_solution[i], graph, initial_seed, nodes_length, budget, num_sim)

    #print('choosen: ' + str(fitness_values[max_index]))
    max_index = np.argmax(final_fitness)

    return final_solution[max_index]


    #new_population = 
    # fitness_mutated_1 = calculate_fitness(mutated_offspring1, graph, initial_seed, nodes_length, budget, num_sim)
    # fitness_mutated_2 = calculate_fitness(mutated_offspring2, graph, initial_seed, nodes_length, budget, num_sim)
    
    # print('fitness mutated 1: ' + str(fitness_mutated_1))
    # print('fitness mutated 2: ' + str(fitness_mutated_2))
    # if fitness_mutated_1 > fitness_mutated_2:
    #     return mutated_offspring1
    # else:
    #     return mutated_offspring2




def random_solution(nodes_length, population_size, graph, initial_seed, budget, num_sim):

    fitness_values = np.zeros(population_size, dtype=int)

    population = list()
    for i in range(population_size):
        balanced_set = np.zeros(2*nodes_length, dtype=int) 

        for j in range(budget):
            index = random.randint(0, 2 * population_size - 1)
            balanced_set[index] = 1

        population.append(balanced_set)
    # for i in range(population_size):
    #     balanced_set = np.random.choice([0, 1], size=2*nodes_length)
    #     population.append(balanced_set)

    for i in range(population_size):
        #def calculate_fitness(population, graph, initial_seed, nodes_length, budget, num_sim):
        fitness_values[i] = calculate_fitness(population[i], graph, initial_seed, nodes_length, budget, num_sim)

    #choose_set_idx = np.argmax(fitness_values)

    generation = 1
    offspring = []
    for gen in range(generation):
        selected_parents = roulette_wheel_selection(population, fitness_values)
        offspring1, offspring2 = one_point_crossover(selected_parents[0], selected_parents[1])
        mutation_probability = 0.01 
        mutated_offspring1 = mutate(offspring1, mutation_probability)
        mutated_offspring2 = mutate(offspring2, mutation_probability)
        offspring.append(mutated_offspring1)
        offspring.append(mutated_offspring2)

    final_solution = population + offspring

    final_fitness = np.zeros(len(final_solution), dtype=int)

    for i in range(len(final_fitness)):
        final_fitness[i] = calculate_fitness(final_solution[i], graph, initial_seed, nodes_length, budget, num_sim)

    max_idx = np.argmax(final_fitness)
    choosen_set = final_solution[max_idx]

    return choosen_set


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--social_network", type=str, help="Path to the social network dataset file")
    parser.add_argument("-i", "--initial_seed_set", type=str, help="Path to the initial seed set file")
    parser.add_argument("-b", "--balanced_seed_set", type=str, help="Path to the balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, help="Budget value")
    args = parser.parse_args()

    # Load the social network, seed sets, and balanced seed sets
    nodes_len, edges_len, social_network = read_social_network(args.social_network)
    initial_seed_set = read_seed_set(args.initial_seed_set)
    #balanced_seed_set = read_seed_set(args.b)
    budget = args.budget

    population_size = 100
    num_sim = 10
    binary_balanced = random_solution(nodes_len, population_size, social_network, initial_seed_set, budget, num_sim)
    #binary_balanced = problem_formulation(nodes_len, population_size, social_network, initial_seed_set, budget, num_sim)
    b1 = 0
    b2 = 0

    # print(binary_balanced)
    # print(len(binary_balanced))
    
    for i in range(nodes_len):
        if binary_balanced[i] == 1:
            b1 = b1 + 1

    for i in range(nodes_len):
        if binary_balanced[nodes_len + i] == 1:
            b2 = b2 + 1


    with open(args.balanced_seed_set, "w") as output_file:
        output_file.write(str(b1) + ' ' + str(b2) + '\n')
        for j in range(nodes_len):
            if binary_balanced[j] == 1:
                output_file.write(str(j) + '\n')
        for j in range(nodes_len):
            if binary_balanced[nodes_len + j] == 1:
                output_file.write(str(j) + '\n')


if __name__ == "__main__":
    main()