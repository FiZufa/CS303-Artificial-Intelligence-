import argparse
import networkx as nx
import random

def read_social_network(file_path):
    social_graph = nx.DiGraph()
    
    with open(file_path, 'r') as file:
        n, m = map(int, file.readline().split())
        
        for _ in range(m):
            u, v, weight_p1, weight_p2 = map(float, file.readline().split())
            
            social_graph.add_edge(int(u), int(v), p1=weight_p1, p2=weight_p2)
    
    return social_graph


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
    node_active = {nd:False for nd in graph.nodes()}
    exposed_nodes = set()
    bfs_queue = list(seed_sets)

    for nd in seed_sets:
        node_active[nd] = True
        exposed_nodes.add(nd)

    while bfs_queue:
        current_node = bfs_queue.pop(0)
        for outward_node in graph.successors(current_node):
            if node_active[outward_node] is False:
                p = edge_prob.get((current_node,outward_node), 0.0)

                if random.random() <= p:
                    node_active[outward_node] = True
                    bfs_queue.append(outward_node)

                exposed_nodes.add(outward_node)

    activated_nodes = len([nd for nd in node_active.values() if nd])
    return activated_nodes, exposed_nodes


def objective_value(num_simulations, social_network, initial_seed, balanced_seed, population):

    combined_c1_seeds = initial_seed['c1_seeds'] | balanced_seed['c1_seeds']
    combined_c2_seeds = initial_seed['c2_seeds'] | balanced_seed['c2_seeds']

    sum = 0

    for i in range(num_simulations):
        activated_c1, exposed_c1 = formulate_diffusion(social_network, combined_c1_seeds, nx.get_edge_attributes(social_network, 'p1'))
        activated_c2, exposed_c2 = formulate_diffusion(social_network, combined_c2_seeds, nx.get_edge_attributes(social_network, 'p2'))

        union_c1 = set(exposed_c1).union(set(balanced_seed['c1_seeds']))
        union_c2 = set(exposed_c2).union(set(balanced_seed['c2_seeds']))

        sym = set(union_c1 - union_c2).union(set(union_c2-union_c1))
        phi = len(set(social_network.nodes).difference(set(sym)))

        sum += phi

    obj_value = sum / num_simulations

    return obj_value






def main():
    parser = argparse.ArgumentParser(description="Information Exposure Maximization")
    parser.add_argument("-n", type=str, help="Path to the social network file")
    parser.add_argument("-i", type=str, help="Path to the initial seed set file")
    parser.add_argument("-b", type=str, help="Path to the balanced seed set file")
    parser.add_argument("-k", type=int, help="Budget")
    parser.add_argument("-o", type=str, help="Path to the output file for objective value")

    args = parser.parse_args()

    # Load the social network, seed sets, and balanced seed sets
    social_network = read_social_network(args.n)
    initial_seed_set = read_seed_set(args.i)
    balanced_seed_set = read_seed_set(args.b)

    n_simulation = 100

    activated_nodes, exposed_nodes = formulate_diffusion(social_network, initial_seed_set['c1_seeds'], nx.get_edge_attributes(social_network, 'p1'))
    objective_val = objective_value(n_simulation, social_network, initial_seed_set, balanced_seed_set)

    print("activated nodes\n")
    print(activated_nodes)
    print("\nexposed_nodes\n")
    print(exposed_nodes)
    #print(initial_seed_set)
    #print(balanced_seed_set)
    #print(initial_seed_set)
    #print(combined_c1_seeds)
    #print(combined_c2_seeds)
    #print(c1_prob)
    #print(c2_prob)
    with open(args.o, "w") as output_file:
        output_file.write(str(objective_val))

    

if __name__ == "__main__":
    main()