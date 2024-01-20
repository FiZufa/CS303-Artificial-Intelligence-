import argparse
import random

# Define a function to read the social network file
def read_social_network(file_path):
    # Implement code to parse the social network file and return relevant data
    social_network = {
        'nodes': set(),
        'edges': [],
        'campaign_weights': {'p1': {}, 'p2': {}}
    }

    with open(file_path, 'r') as file:
        # Read the first line to get the number of nodes and edges
        n, m = map(int, file.readline().split())

        for _ in range(m):
            u, v, weight_p1, weight_p2 = map(float, file.readline().split())

            # Add nodes to the set of nodes
            social_network['nodes'].add(u)
            social_network['nodes'].add(v)

            # Add edge information to the list of edges and campaign weights
            social_network['edges'].append((u, v))
            social_network['campaign_weights']['p1'][(u, v)] = weight_p1
            social_network['campaign_weights']['p2'][(u, v)] = weight_p2

    return social_network


# Define a function to read the seed set file
def read_seed_set(file_path):
    # Implement code to parse the seed set file and return relevant data
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

# Define a function to compute the objective value
def compute_objective_value(social_network, initial_seed_set, balanced_seed_set):
    # Implement code to compute the objective value based on the provided data
    # You'll use the data from the social network and seed sets to perform calculations
    # Return the computed objective value
    nodes = social_network['nodes']
    edges = social_network['edges']
    campaign_weights = social_network['campaign_weights']

    k1 = len(initial_seed_set['c1_seeds'])
    k2 = len(initial_seed_set['c2_seeds'])

    # Initialize sets to track reached nodes for each campaign
    reached_c1 = set(initial_seed_set['c1_seeds'])
    reached_c2 = set(initial_seed_set['c2_seeds'])

    # Implement the Independent Cascade model to simulate information propagation
    # Iterate until no more nodes can be activated
    while True:
        new_reached_c1 = set()
        new_reached_c2 = set()

        for u, v in edges:
            if u in reached_c1 and v not in reached_c1:
                # Node u tries to activate node v for campaign c1
                p = campaign_weights['p1'][(u, v)]
                if random.random() <= p:
                    new_reached_c1.add(v)

            if u in reached_c2 and v not in reached_c2:
                # Node u tries to activate node v for campaign c2
                p = campaign_weights['p2'][(u, v)]
                if random.random() <= p:
                    new_reached_c2.add(v)

        if not new_reached_c1 and not new_reached_c2:
            break

        # Update the reached sets
        reached_c1.update(new_reached_c1)
        reached_c2.update(new_reached_c2)

    # Calculate the objective value
    common_reached_nodes = reached_c1.intersection(reached_c2)
    unexposed_nodes = nodes.difference(reached_c1.union(reached_c2))
    objective_value = len(common_reached_nodes) + len(unexposed_nodes)

    return objective_value

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Information Exposure Maximization")
    parser.add_argument("-n", type=str, help="Path to the social network file")
    parser.add_argument("-i", type=str, help="Path to the initial seed set file")
    parser.add_argument("-b", type=str, help="Path to the balanced seed set file")
    parser.add_argument("-k", type=int, help="Budget")
    parser.add_argument("-o", type=str, help="Path to the output file for objective value")

    args = parser.parse_args()

    # Read input data from files using the functions you've implemented
    social_network = read_social_network(args.n)
    initial_seed_set = read_seed_set(args.i)
    balanced_seed_set = read_seed_set(args.b)

    # print(initial_seed_set)
    # print('\n{}'.format(balanced_seed_set))

    # Call the compute_objective_value function with the retrieved data
    #objective_value = compute_objective_value(social_network, initial_seed_set, balanced_seed_set)

    num_runs = 10000  # Number of times to run the simulation
    objective_values = []

    for _ in range(num_runs):
        objective_value = compute_objective_value(social_network, initial_seed_set, balanced_seed_set)
        objective_values.append(objective_value)

    # Calculate the average objective value
    average_objective_value = sum(objective_values) / num_runs

    # print("Average Objective Value:", average_objective_value)
    # for i in range(10):
    #     print(social_network[i])
    #     print("\n")
        
    # Write the objective value to the output file
    with open(args.o, "w") as output_file:
        output_file.write(str(average_objective_value))

if __name__ == "__main__":
    main()
