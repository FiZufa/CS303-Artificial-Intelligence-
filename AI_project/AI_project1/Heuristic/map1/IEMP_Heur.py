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

# def formulate_diffusion(graph, seed_sets, edge_prob):
#     node_active = {nd:False for nd in graph.nodes()}
#     exposed_nodes = set()
#     bfs_queue = list(seed_sets)

#     for nd in seed_sets:
#         node_active[nd] = True
#         exposed_nodes.add(nd)

#     while bfs_queue:
#         current_node = bfs_queue.pop(0)
#         for outward_node in graph.successors(current_node):
#             if node_active[outward_node] is False:
#                 p = edge_prob.get((current_node,outward_node), 0.0)

#                 if random.random() <= p:
#                     node_active[outward_node] = True
#                     bfs_queue.append(outward_node)

#                 exposed_nodes.add(outward_node)

#     activated_nodes = len([nd for nd in node_active.values() if nd])
#     return activated_nodes, exposed_nodes


def formulate_diffusion(graph, seed_sets, edge_prob):
    node_active = {nd:False for nd in graph.nodes()}
    exposed_nodes = {nd:False for nd in graph.nodes()}
    bfs_queue = list(seed_sets)

    for nd in seed_sets:
        node_active[nd] = True
        exposed_nodes[nd] = True
    
    while bfs_queue:
        current_node = bfs_queue.pop(0)
        for outward_node in graph.successors(current_node):
            if node_active[outward_node] is False:
                p = edge_prob.get((current_node, outward_node), 0.0)

                if random.random() <= p:
                    node_active[outward_node] = True
                    bfs_queue.append(outward_node)

                exposed_nodes[outward_node] = True

    activated_nodes_num = len([nd for nd in node_active.values() if nd])
    return activated_nodes_num, node_active, exposed_nodes


def objective_value(num_simulations, social_network, initial_seed, S1, S2):
    combined_c1_seeds = initial_seed['c1_seeds'] | S1
    combined_c2_seeds = initial_seed['c2_seeds'] | S2

    sum = 0

    for i in range(num_simulations):
        activated_c1_num, node_active_c1, exposed_c1 = formulate_diffusion(social_network, combined_c1_seeds, nx.get_edge_attributes(social_network, 'p1'))
        activated_c2_num, node_active_c2, exposed_c2 = formulate_diffusion(social_network, combined_c2_seeds, nx.get_edge_attributes(social_network, 'p2'))

        union_c1 = set(exposed_c1).union(set(S1))
        union_c2 = set(exposed_c2).union(set(S2))

        sym = set(union_c1 - union_c2).union(set(union_c2 - union_c1))
        phi = len(set(social_network.nodes).difference(set(sym)))

        sum += phi

    obj_value = sum / num_simulations

    return obj_value

# def calculate_gain(num_simulations, social_network, initial_seed, S1, new_S1, S2):
#     phi_before = objective_value(num_simulations, social_network, initial_seed, S1, S2)
#     phi_after = objective_value(num_simulations, social_network, initial_seed, new_S1, S2)
#     gain = phi_after - phi_before

#     return gain

# def greedy_best_first_search(num_simulations, social_network, initial_seed, budget):
#     S1 = set()
#     S2 = set()

#     while len(S1) + len(S2) <= budget:
#         best_gain_1 = float('-inf')
#         best_gain_2 = float('-inf')
#         v1_star = None
#         v2_star = None

#         for v in social_network.nodes():
#             if v not in S1:
#                 new_S1 = S1.union({v})
#                 gain_1 = calculate_gain(num_simulations, social_network, initial_seed, S1, new_S1, S2)
#                 if gain_1 > best_gain_1:
#                     best_gain_1 = gain_1
#                     v1_star = v

#             if v not in S2:
#                 new_S2 = S2.union({v})
#                 gain_2 = calculate_gain(num_simulations, social_network, initial_seed, S1, new_S2, S2)
#                 if gain_2 > best_gain_2:
#                     best_gain_2 = gain_2
#                     v2_star = v
            
#             if v1_star and v2_star:
#                 if best_gain_1 > best_gain_2: 
#                     S1.add(v1_star)
#                 else:
#                     S2.add(v2_star)

#             elif v1_star:
#                 S1.add(v1_star)
#             elif v2_star:
#                 S2.add(v2_star)

#     return S1, S2


def increment_set(graph, is_active_set, is_exposed_set):
    active_increment, exposed_increment = set()

    return active_increment, exposed_increment


def greedy_bfs(N, graph, initial_seed, k):
    S1, S2 = set()
    total_gain_c1 = 0
    total_gain_c2 = 0

    prob_c1 = nx.get_edge_attributes(graph, 'p1')
    prob_c2 = nx.get_edge_attributes(graph, 'p2')

    while len(S1) + len(S2) <= k :

        # N monte carlo simulation
        for n in range(N):

            combined_c1 = initial_seed['c1_seeds'] | S1
            combined_c2 = initial_seed['c2_seeds'] | S2
            # find active and exposed set for each seed set
            num_c1, is_active_c1, is_exposed_c1 = formulate_diffusion(graph, combined_c1, prob_c1)
            num_c2, is_active_c2, is_exposed_c2 = formulate_diffusion(graph, combined_c2, prob_c2)
            # for each v in G
            for v in graph.nodes():
                active_increment_c1, exposed_increment_c1 = increment_set(graph, is_active_c1, is_exposed_c1)
                active_increment_c2, exposed_increment_c2 = increment_set(graph, is_active_c2, is_exposed_c2)

                # 𝒉𝟏_𝒋(𝒗_𝒊  ) = 𝚽(𝑺_𝟏∪{𝒗_𝒊 }, 𝑺_𝟐 )−𝚽(𝑺_𝟏, 𝑺_𝟐 )
                # 𝒉𝟐_𝒋(𝒗_𝒊  ) = 𝚽(𝑺_𝟏, 𝑺_𝟐∪{𝒗_𝒊 })−𝚽(𝑺_𝟏, 𝑺_𝟐 )
                phi_S1_S2 = objective_value(1, graph, initial_seed, S1, S2)
                phi_newS1_S2 = objective_value(1, graph, initial_seed, S1 | exposed_increment_c1, S2)
                phi_S1_newS2 = objective_value(1, graph, initial_seed, S1, S2 | exposed_increment_c2)

                h1 = phi_newS1_S2 - phi_S1_S2
                h2 = phi_S1_newS2 - phi_S1_S2
                #h1 = calculate_gain(1, graph, initial_seed, S1, exposed_increment_c1, S2)
                #h2 = calculate_gain(1, graph, initial_seed, S1, exposed_increment_c2, S2)

            total_h1_c1 += h1 # this is a number
            total_h2_c2 += h2 # this is a number

        avg_gain_c1 = total_gain_c1 / N
        avg_gain_c2 = total_gain_c2 / N

        # end then append v_1_star and v_2_star to balanced seed node, but how?




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, help="The absolute path of the social network file")
    parser.add_argument("-i", type=str, help="The absolute path of the two campaigns' initial seed set")
    parser.add_argument("-b", type=str, help="The absolute path of the two campaigns' balanced seed set")
    parser.add_argument("-k", type=int, help="The positive integer budget")
    args = parser.parse_args()
    n = args.n
    i = args.i
    b = args.b
    k = args.k


    social_network = read_social_network(n)
    initial_seed_set = read_seed_set(i)
    # budget = k

    # num_simulations = 1
    print(social_network)
    #print(social_network[0])
    print(initial_seed_set)


    #S1, S2 = greedy_best_first_search(num_simulations, social_network, initial_seed_set, budget)
    #print_balanced_seed_set(S1, S2, args.b)

    
    # with open(b, 'w') as f:
    #     f.write(str(k-2) + ' ' + str(2)+'\n')
    #     for i in range(k):
    #         f.write(str(i+1) + '\n')

    # with open(b, 'w') as f:
    #     f.write(str(len(S1)) + ' ' + str(len(S2))+'\n')
    #     for i in S1:
    #         f.write(str(i) + '\n')
    #     for i in S2:
    #         f.write(str(i) + '\n')


