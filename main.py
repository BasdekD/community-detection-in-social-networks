import utilities
import networkx as nx
from time import perf_counter

t1_start = perf_counter()


# IMPORT DATASET (UNDIRECTED, UNWEIGHTED GRAPH WITH NON-OVERLAPPING COMMUNITIES)

# TEST GRAPH
# graph = utilities.loadDummyDataset()

# REAL DATASET GRAPH
print("Loading Dataset...")
graph = utilities.loadDataset("football\\football-edges.txt")

# GETTING LIST OF LOCAL CLUSTERING COEFFICIENT FOR EACH NODE
clustering_coefficient = {}
for node in graph.nodes:
    print("Calculating CC of node {}...".format(node))
    clustering_coefficient[node] = utilities.getLocalClusteringCoefficient(graph, node)

# GETTING THE MOST SIMILAR NODES
print("Getting most similar nodes...")
most_similar_nodes, isolated_node_list = utilities.getMostSimilarNodes(graph, clustering_coefficient)

# GETTING LEGIT COMBINATIONS (LIMIT = 100)
print("Getting possible combinations...")
connected_comp = utilities.getLegitCombinations(most_similar_nodes, 100)

# REMOVING DUPLICATES FROM CONNECTED COMPONENTS
print("Removing duplicate connected components...")
unique_connected_comp = utilities.getUniqueConnectedComponents(connected_comp)

# GETTING BASIC COMMUNITIES FROM CONNECTED COMPONENTS
all_possible_basic_communities = []
for component in unique_connected_comp:
    print("Extracting basic community from component {}...".format(component))
    all_possible_basic_communities.append(utilities.getBasicCommunities(component))

# APPLYING MERGING STRATEGY AND GETTING THE HIGHEST MODULARITY PARTITION OF THE GRAPH
threshold = int(len(graph.nodes)/2)
max_modularity = 0
best_partition = []
best_threshold = 0
for i in range(1, threshold+1):
    print("Applying merging strategy for threshold {}...".format(i))
    optimized_communities = []
    for basic_community in all_possible_basic_communities:
        optimized_comm_structure = utilities.mergingStrategy(graph, basic_community, i)
        communities = []
        for community in optimized_comm_structure:
            take_nodes = nx.Graph()
            take_nodes.add_edges_from(community)
            community_nodes = list(take_nodes.nodes)
            communities.append(set(community_nodes))
        if isolated_node_list:
            for node in isolated_node_list:
                communities.append({node})
        print(communities)
        modularity = nx.algorithms.community.modularity(graph, communities)
        if modularity > max_modularity:
            max_modularity = modularity
            best_partition = communities
            best_threshold = i

utilities.printResults(best_partition, max_modularity, best_threshold)
for cluster in best_partition:
    for node in cluster:
        graph.nodes[node]['label'] = best_partition.index(cluster)
print(graph.nodes('label'))

t1_stop = perf_counter()
print("Elapsed time:", t1_stop-t1_start)


