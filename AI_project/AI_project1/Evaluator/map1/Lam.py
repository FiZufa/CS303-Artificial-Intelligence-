import argparse
import networkx as nx
import random


def load_dataset1(file_path):
    social_network = nx.DiGraph()  # Create a directed graph
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            node1, node2, prob_campaign1, prob_campaign2 = map(float, parts[:4])
            social_network.add_edge(int(node1), int(node2), prob_campaign1=prob_campaign1,
                                    prob_campaign2=prob_campaign2)
    return social_network


def load_seed(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        A, B = map(int, lines[0].split())
        seed_set_1 = {int(line.strip()) for line in lines[1: 1 + A]}
        seed_set_2 = {int(line.strip()) for line in lines[1 + A:]}
    return seed_set_1, seed_set_2


def load_seed_balance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        A, B = map(int, lines[0].split())
        balanced_seed_set_1 = {int(line.strip()) for line in lines[1: 1 + A]}
        balanced_seed_set_2 = {int(line.strip()) for line in lines[1 + A:]}
    return balanced_seed_set_1, balanced_seed_set_2


def simulate_diffusion(network, initial_seed_set, campaign_probabilities):
    is_activated = {node: False for node in network.nodes()}
    exposed_nodes = set()
    queue = list(initial_seed_set)

    for node in initial_seed_set:
        is_activated[node] = True
        exposed_nodes.add(node)

    while queue:
        current_node = queue.pop(0)
        for neighbor in network.successors(current_node):  # Consider outgoing edges
            if not is_activated[neighbor]:
                activation_prob = random.random()
                campaign_probability = campaign_probabilities.get((current_node, neighbor), 0.0)

                if activation_prob < campaign_probability:
                    is_activated[neighbor] = True
                    queue.append(neighbor)

                exposed_nodes.add(neighbor)

    activated_nodes = sum(1 for node in is_activated.values() if node)
    return activated_nodes, exposed_nodes


def main():
    parser = argparse.ArgumentParser(description="Evaluator for Balanced Seed Sets")
    parser.add_argument("-n", "--social_network", type=str, help="Path to the social network dataset file")
    parser.add_argument("-i", "--initial_seed_set", type=str, help="Path to the initial seed set file")
    parser.add_argument("-b", "--balanced_seed_set", type=str, help="Path to the balanced seed set file")
    parser.add_argument("-k", "--budget", type=int, help="Budget value")
    parser.add_argument("-o", "--object_value_output_path", type=str, help="Path to save the objective value output")

    args = parser.parse_args()

    # Load the social network, seed sets, and balanced seed sets
    social_network = load_dataset1(args.social_network)
    seed_set_campaign1, seed_set_campaign2 = load_seed(args.initial_seed_set)
    balanced_seed_set_campaign1, balanced_seed_set_campaign2 = load_seed_balance(args.balanced_seed_set)

    # print(social_network)

    # Generate random edge activation probabilities for campaigns (you should replace this with your actual data)
    campaign1_probabilities = nx.get_edge_attributes(social_network, 'prob_campaign1')
    campaign2_probabilities = nx.get_edge_attributes(social_network, 'prob_campaign2')

    # Simulate the diffusion process with the initial seed set
    num_simulations = 1000  # Number of Monte Carlo simulations

    # Φ(𝑆1, 𝑆2) = 𝔼[|V∖(r1(I1 ⋃ S1) △ r2(I2 ⋃ S2))|]
    total = 0
    for _ in range(num_simulations):
        activated_nodes1, exposed_nodes1 = simulate_diffusion(social_network, seed_set_campaign1 | balanced_seed_set_campaign1,
                                                              campaign1_probabilities)
        activated_nodes2, exposed_nodes2 = simulate_diffusion(social_network, seed_set_campaign2 | balanced_seed_set_campaign2,
                                                              campaign2_probabilities)

        union1 = set(exposed_nodes1).union(set(balanced_seed_set_campaign1))
        union2 = set(exposed_nodes2).union(set(balanced_seed_set_campaign2))

        symmetric_difference = set(union1 - union2).union(set(union2 - union1))

        _phi = len(set(social_network.nodes).difference(set(symmetric_difference)))

        total += _phi

    objective_value = total / num_simulations

    # Save the objective value to the specified output file
    with open(args.object_value_output_path, "w") as objective_value_file:
        objective_value_file.write(f"{objective_value}\n")


if __name__ == "__main__":
    main()
