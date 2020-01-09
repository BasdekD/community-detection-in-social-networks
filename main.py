import utilities
import networkx as nx

# IMPORT DATASET (UNDIRECTED, UNWEIGHTED GRAPH WITH NON-OVERLAPPING COMMUNITIES)

# TEST GRAPH
# graph = utilities.loadDummyDataset()

# REAL DATASET GRAPH
graph = utilities.loadDataset("polbooks\\polbooks-edges.txt")


all_similarities = []
max_similarities = []
most_similar_nodes = []

# GETTING SIMILARITIES OF ADJACENT NODES
for node in graph.nodes:
    print("Calculating Similarities for node" + str(node) + "...")
    similarity_per_node = []
    for neighbor in graph.neighbors(node):
        similarity_per_node.append(utilities.similarityIndex(graph, node, neighbor))
    all_similarities.append(similarity_per_node)

# GETTING THE MAXIMUM SIMILARITY FOR EACH NODE
for similarity_list in all_similarities:
    max_similarities.append(max(similarity_list))

# GETTING THE LIST OF MOST SIMILAR NODES PAIRS
index = 0
for node in graph:
    for neighbor in graph.neighbors(node):
        if utilities.similarityIndex(graph, node, neighbor) >= max_similarities[index]:
            most_similar_nodes.append((node, neighbor))
    index += 1

connected_comp = utilities.getLegitCombinations(most_similar_nodes, 100)
unique_connected_comp = utilities.getUniqueConnectedComponents(connected_comp)


all_possible_basic_communities = []

for component in unique_connected_comp:
    all_possible_basic_communities.append(utilities.getBasicCommunities(component))

threshold = int(len(graph.nodes)/2)
max_modularity = 0
best_partition = []
best_threshold = 0
for i in range(1, threshold+1):
    optimized_communities = []
    for basic_community in all_possible_basic_communities:
        optimized_comm_structure = utilities.mergingStrategy(graph, basic_community, i)
        communities = []
        for community in optimized_comm_structure:
            take_nodes = nx.Graph()
            take_nodes.add_edges_from(community)
            community_nodes = list(take_nodes.nodes)
            communities.append(set(community_nodes))
        modularity = nx.algorithms.community.modularity(graph, communities)
        if modularity > max_modularity:
            max_modularity = modularity
            best_partition = communities
            best_threshold = i

print("Number of communities: " + str(len(best_partition)))
print(best_partition)
print(max_modularity)
print(best_threshold)
