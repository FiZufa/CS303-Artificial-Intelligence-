import networkx as nx
import matplotlib.pyplot as plt


def vis(colors, position, G):
    fig, ax = plt.subplots()
    edge_labels = dict([((u, v), d['weight']) for u, v, d in G.edges(data=True)]) # dictionary with key (u,v)
    nx.draw(G, pos=position, with_labels=True, node_color=colors)  # get positions
    nx.draw_networkx_edge_labels(G, position, edge_labels=edge_labels)  # draw edge
    #ax.xaxis.set_major_locator(plt.NullLocator())  # delete x axis
    #ax.yaxis.set_major_locator(plt.NullLocator())  # delete y axis
    plt.savefig("pic2.png")

node_names = ['A', 'T', 'Z', 'O', 'L', 'D', 'S', 'R', 'C', 'F', 'P', 'B']
distances_to_end = [421, 409, 432, 435, 357, 171, 215, 156, 183, 140, 108, 0]

# Create a dictionary to store distances from each node to the end node
distances_dict = {node: distance for node, distance in zip(node_names, distances_to_end)}

# Create an adjacency list to represent the graph
graph = {
    'A': [('Z', 75), ('S', 140), ('T', 118)],
    'Z': [('O', 71)],
    'O': [('S', 151)],
    'T': [('L', 70)],
    'L': [('D', 145)],
    'D': [('C', 120)],
    'S': [('R', 80), ('F', 99)],
    'R': [('P', 97), ('C', 146)],
    'C': [('P', 138)],
    'F': [('B', 211)],
    'P': [('B', 101)],
    'B': []
}

node_positions = {
    'A': (-1.9, 0.9),
    'T': (-1.9, -0.3),
    'Z': (-1.9, -1.9),
    'O': (-0.3, 0.9),
    'L': (-0.3, -0.3),
    'D': (-0.3, -1.9),
    'S': (0.9, 0.9),
    'R': (0.9, -0.3),
    'C': (0.9, -1.9),
    'F': (2.1, 0.9),
    'P': (2.1, -0.3),
    'B': (2.1, -1.9)
}

colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'yellow', 'brown', 'gray', 'black']

G = nx.Graph()

for node, edges in graph.items():
    for edge in edges:
        neighbor, weight = edge #(neighbor, weight)
        G.add_edge(node, neighbor, weight=weight)


vis(colors, node_positions, G)

"""
def AStarSearch(Graph,start,end,distances):

    queue = []
    # TODO: write your code :)
    # Initialize queue here
    #while (len(queue) > 0):
    yield queue  # yield queue whenever before an element popped out from the queue
        # TODO: write your code :)
        # write your algorithm
"""

def AStarSearch(G, start, end, distances_dict):

    queue = [(0, start)] # (f, node)
    explored = set() 

    g_value = {node: float('inf') for node in G} # node : g(n)
    g_value[start] = 0
    parent = {start: None}

    while queue:

        queue.sort() # minheap-> smallest f(n) = f(n) + g(n)
        current_f, current_node = queue.pop(0)

        if current_node == end: # arrive destination

            path = [] # path from start
            while current_node:
                path.insert(0, current_node)
                current_node = parent[current_node]
            yield path

        else:
            for neighbor in G[current_node]:
                if isinstance(neighbor, tuple) and len(neighbor) == 2:
                    neighbor, weight = neighbor
                else:
                # Handle the case where neighbor is not a tuple with (neighbor, weight)
                    weight = 0  # Provide a default weight or handle it differently if needed
                    continue

                tentative_g = g_value[current_node] + weight  # Access weight directly

                if tentative_g < g_value[neighbor]:
                    g_value[neighbor] = tentative_g
                    parent[neighbor] = current_node

                    h_node = distances_dict[neighbor]  # Use distances_dict to look up distances
                    f_value = tentative_g + h_node

                    queue.append((f_value, neighbor))

        explored.add(current_node)

    return



for path in AStarSearch(G,node_names[0], node_names[-1], distances_dict):
    print(path)


# test block
test_case = 1
G = nx.DiGraph()  # for visualization
position = {}
result = []

# read file
distances={}
with open(f'./test_cases/{test_case}.txt', 'r') as f:
    line = f.readline()
    all_nodes = line.strip().split(" ")
    line = f.readline()
    dis=line.strip().split(" ")
    for i in range(len(all_nodes)):
        distances[all_nodes[i]]=float(dis[i])
    line=f.readline()
    for i in range(int(line)):
        line = f.readline()
        edge = line.strip().split(" ")
        G.add_edge(edge[0], edge[1], weight=float(edge[2]))
    pos = f.readline().strip().split(" ")
    for i in range(len(all_nodes)):
        position[all_nodes[i]] = (float(pos[i * 2]), float(pos[2 * i + 1]))
Graph = dict([(u, []) for u, v, d in G.edges(data=True)])
for u, v, d in G.edges(data=True):
    Graph[u].append((v, d["weight"]))
for node in G:
    if node not in Graph.keys():
        Graph[node]=[]
# Visualization
gray = (0.5, 0.5, 0.5)
brown = (0.5, 0.25, 0)
white = (1, 1, 1)
colors_list = [(_i, white) for _i in G.nodes]
colors_dict = dict(colors_list)
start=all_nodes[0]
end=all_nodes[-1]
res = AStarSearch(Graph,start,end,distances)
q = next(res)
temp_q=[]
last_q=q.copy()
last_node = None
while True:
    try:
        for node in G.nodes:
            if node in q and colors_dict[node] == white:
                colors_dict[node] = brown
            elif node not in q and colors_dict[node] == brown:
                colors_dict[node] = gray
                result.append(node)
        nodes, colors = zip(*colors_dict.items())
        vis(colors, position, G)
        if white not in colors:
            last_node = q[0]
        q = next(res)
        temp_q=last_q.copy()
        last_q=q.copy()
        if end in temp_q and end not in q:
            last_node=end
            break
    except StopIteration:
        break
for node in G.nodes:
    if node == last_node:
        colors_dict[node] = gray
result.append(last_node)
nodes, colors = zip(*colors_dict.items())
vis(colors, position, G)

plt.savefig("graph_plot.png")
print(result)