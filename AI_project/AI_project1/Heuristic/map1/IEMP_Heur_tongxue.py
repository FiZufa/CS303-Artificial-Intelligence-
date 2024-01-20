import argparse
import random
import time
import copy
import numpy as np

start = time.perf_counter()

parser = argparse.ArgumentParser(description='Heuristic Algorithm')
parser.add_argument('-n', type=str, help='Path of the social network file')
parser.add_argument('-i', type=str, help='Path of the two campaigns initial seed set')
parser.add_argument('-b', type=str, help='Path of the the two campaigns balanced seed set')
parser.add_argument('-k', type=int, help='The positive integer budget')

args = parser.parse_args()


def read_network(network_file):
    network = {}
    with open(network_file, 'r') as f:
        line = f.readline()
        line = line.strip().split()
        n = int(line[0])
        m = int(line[1])
        for i in range(n):
            network[i] = []
        for i in range(m):
            line = f.readline()
            line = line.strip().split()
            u = int(line[0])
            v = int(line[1])
            w1 = float(line[2])
            w2 = float(line[3])
            network[u].append((v, w1, w2))
    return network


def read_seed_set(seed_set_file):
    seed_set_A = []
    seed_set_B = []
    with open(seed_set_file, 'r') as f:
        line = f.readline()
        line = line.strip().split()
        num_seed_A = int(line[0])
        num_seed_B = int(line[1])
        for i in range(num_seed_A):
            line = f.readline()
            line = line.strip().split()
            seed_set_A.append(int(line[0]))
        for i in range(num_seed_B):
            line = f.readline()
            line = line.strip().split()
            seed_set_B.append(int(line[0]))
    return seed_set_A, seed_set_B


network = read_network(args.n)
res_activate = read_seed_set(args.i)
budget = args.k

print(network[0])
print(res_activate)
# pathn = 'Heuristic/map3/dataset2'
# pathi = 'Heuristic/map3/seed2'
# pathb = 'Heuristic/map3/answer'
# network = read_network(pathn)
# res_activate = read_seed_set(pathi)
# budget = 15

n = len(network)

rates = {}
for node in network: #for each node in
    rates[node] = 0
    for edge in network[node]:
        rates[node] += edge[1] + edge[2] # list edge p1 + p2
rates = sorted(rates.items(), key=lambda item: item[1], reverse=True)
# sort the nodes based on what the prob
# rate dictionary: rate[node][prob]

selected_nodes = []
if n < 500:
    for i in range(10):
        selected_nodes.append(rates[i][0]) #select 10 nodes with 
else:
    for i in range(10 * budget):
        selected_nodes.append(rates[i][0])

# print(selected_nodes)
# print('rate 0: ' + str(rates[0][1]))
#print("node 0: " + str(selected_nodes[0][0]) )
seed_set_A = res_activate[0]
seed_set_B = res_activate[1]


def simulate_A(activate_input):
    activate = activate_input.copy()
    exposed = activate.copy()
    queue = activate.copy()
    while queue:
        head = queue.pop(0)
        for edge in network[head]:
            if edge[0] not in exposed:
                exposed.append(edge[0])
            if edge[0] not in activate:
                rate = random.random()
                if rate < edge[1]:
                    activate.append(edge[0])
                    queue.append(edge[0])
    return activate, exposed


def simulate_B(activate_input):
    activate = activate_input.copy()
    exposed = activate.copy()
    queue = activate.copy()
    while queue:
        head = queue.pop(0)
        for edge in network[head]:
            if edge[0] not in exposed:
                exposed.append(edge[0])
            if edge[0] not in activate:
                rate = random.random()
                if rate < edge[2]:
                    activate.append(edge[0])
                    queue.append(edge[0])
    return activate, exposed


def CalculateInfluence(network, seed_set_A_input, seed_set_B_input):
    numOfSimulations = 5
    number = 0

    for i in range(numOfSimulations):
        seed_set_A = seed_set_A_input.copy()
        seed_set_B = seed_set_B_input.copy()

        Explored_A = seed_set_A.copy()
        Explored_B = seed_set_B.copy()

        queue_A = seed_set_A.copy()
        queue_B = seed_set_B.copy()
        # for node in seed_set_A:

        while queue_A:
            head = queue_A.pop(0)

            for edge in network[head]:
                if edge[0] not in Explored_A:
                    Explored_A.append(edge[0])

                if edge[0] not in seed_set_A:
                    rate = random.random()

                    if rate < edge[1]:
                        seed_set_A.append(edge[0])
                        queue_A.append(edge[0])

        # for node in seed_set_B:
        while queue_B:
            head = queue_B.pop(0)
            for edge in network[head]:
                if edge[0] not in Explored_B:
                    Explored_B.append(edge[0])
                if edge[0] not in seed_set_B:
                    rate = random.random()
                    if rate < edge[2]:
                        seed_set_B.append(edge[0])
                        queue_B.append(edge[0])
        
        # n = the length of the graph
        number += n - len(set(Explored_A) ^ set(Explored_B))
    return number / numOfSimulations


balanced_set_A = []
balanced_set_B = []

simulate_times = 10

answer_A = np.zeros((simulate_times, len(selected_nodes)), dtype=int)
answer_B = np.zeros((simulate_times, len(selected_nodes)), dtype=int)

while len(balanced_set_A) + len(balanced_set_B) < budget:
    # start1 = time.perf_counter()
    for i in range(simulate_times):
        activate_A = seed_set_A.copy() + balanced_set_A
        activate_A_simulate, exposed_A_simulate = simulate_A(activate_A)

        activate_B = seed_set_B.copy() + balanced_set_B
        activate_B_simulate, exposed_B_simulate = simulate_B(activate_B)
        # activate_A_calAnswer = activate_A_simulate.copy()
        # activate_B_calAnswer = activate_B_simulate.copy()
        # answer_origin = CalculateInfluence(network, activate_A_calAnswer, activate_B_calAnswer)
        # for node in selected_nodes:
        for j in range(len(selected_nodes)):
            node = selected_nodes[j]

            answer_A_sim, answer_B_sim = 0, 0
            queue_A = [node]

            activate_A_simNode = activate_A_simulate.copy()
            exposed_A_simNode = exposed_A_simulate.copy()

            if node not in activate_A_simNode:
                activate_A_simNode.append(node)

                if node not in exposed_A_simNode:
                    exposed_A_simNode.append(node)

                while queue_A:
                    head = queue_A.pop(0)
                    for edge in network[head]:
                        if edge[0] not in exposed_A_simNode:
                            exposed_A_simNode.append(edge[0])
                        if edge[0] not in activate_A_simNode:
                            rate = random.random()
                            if rate < edge[1]:
                                activate_A_simNode.append(edge[0])
                                queue_A.append(edge[0])
                answer_A_sim = n - len(set(exposed_A_simNode) ^ set(exposed_B_simulate))

            queue_B = [node]
            activate_B_simNode = activate_B_simulate.copy()
            exposed_B_simNode = exposed_B_simulate.copy()

            if node not in activate_B_simNode:
                activate_B_simNode.append(node)
                if node not in exposed_B_simNode:
                    exposed_B_simNode.append(node)
                while queue_B:
                    head = queue_B.pop(0)
                    for edge in network[head]:
                        if edge[0] not in exposed_B_simNode:
                            exposed_B_simNode.append(edge[0])
                        if edge[0] not in activate_B_simNode:
                            rate = random.random()
                            if rate < edge[2]:
                                activate_B_simNode.append(edge[0])
                                queue_B.append(edge[0])
                answer_B_sim = n - len(set(exposed_B_simNode) ^ set(exposed_A_simulate))

            answer_A[i][j] = answer_A_sim
            answer_B[i][j] = answer_B_sim
            
    answer_A_mean = np.mean(answer_A, axis=0).tolist()
    answer_B_mean = np.mean(answer_B, axis=0).tolist()
    if max(answer_A_mean) > max(answer_B_mean):
        answer_A_max_node = answer_A_mean.index(max(answer_A_mean))
        balanced_set_A.append(selected_nodes[answer_A_max_node])
    else:
        answer_B_max_node = answer_B_mean.index(max(answer_B_mean))
        balanced_set_B.append(selected_nodes[answer_B_max_node])

    # end1 = time.perf_counter()
    # print('Running time: %s Seconds' % (end1 - start1))

# with open(args.b, 'w') as f:
#     f.write(str(len(balanced_set_A)) + ' ' + str(len(balanced_set_B)) + '\n')
#     for node in balanced_set_A:
#         f.write(str(node) + '\n')
#     for node in balanced_set_B:
#         f.write(str(node) + '\n')

# with open(pathb, 'w') as f:
#     f.write(str(len(balanced_set_A)) + ' ' + str(len(balanced_set_B)) + '\n')
#     for node in balanced_set_A:
#         f.write(str(node) + '\n')
#     for node in balanced_set_B:
#         f.write(str(node) + '\n')

end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))
