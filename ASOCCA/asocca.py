import ASOCCA.utilities as utilities
import networkx as nx


def main():
    """
    Implementation of a non-overlapping community detection algorithm named ASOCCA introduced in
    "Pan, X., Xu, G., Wang, B., & Zhang, T. (2019). A Novel Community Detection Algorithm Based on Local Similarity of
    Clustering Coefficient in Social Networks. IEEE Access, 7, 121586-121598. doi:10.1109/access.2019.2937580"

    :return: The community structure that optimizes the modularity metric, the optimal modularity value
    """

    # Uncomment the line below to load a toy dataset (you must comment the line loading the real dataset)
    # graph = utilities.loadDummyDataset()

    print("Loading Dataset...")
    # Load a dataset available in the dataset folder of the project
    graph = utilities.loadDataset("football-edges.txt")

    # Get a list with the local clustering coefficient value of each node in the graph
    clustering_coefficient = {}
    print("Calculating CC of each node...")
    for node in graph.nodes:
        clustering_coefficient[node] = utilities.getLocalClusteringCoefficient(graph, node)

    # Get pairs of the most similar nodes based on the Similarity Index defined in the paper cited at the top of this
    # file
    print("Getting most similar nodes...")
    most_similar_nodes, isolated_node_list = utilities.getMostSimilarNodes(graph, clustering_coefficient)

    # Get legit combinations of similar pairs of nodes (limit is 100 for computational reasons as proposed in the
    # paper
    print("Getting possible combinations...")
    connected_comp = utilities.getLegitCombinations(most_similar_nodes, 100)

    # Remove duplicates from the connected components that have occurred
    print("Removing duplicate connected components...")
    unique_connected_comp = utilities.getUniqueConnectedComponents(connected_comp)

    # Find all possible community structures based upon the connected components
    all_possible_basic_communities = []
    print("Extracting basic community from components...")
    for component in unique_connected_comp:
        all_possible_basic_communities.append(utilities.getBasicCommunities(component))

    # Apply the merging strategy proposed in the paper and keep the community structure that maximizes the value of
    # modularity index
    threshold = int(len(graph.nodes)/2)
    max_modularity = 0
    best_partition = []
    best_threshold = 0
    print("Applying merging strategy for different threshold values...")
    for i in range(1, threshold+1):
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
            modularity = nx.algorithms.community.modularity(graph, communities)
            if modularity > max_modularity:
                max_modularity = modularity
                best_partition = communities
                best_threshold = i

    utilities.printResults(best_partition, max_modularity, best_threshold)
    # for cluster in best_partition:
    #     for node in cluster:
    #         graph.nodes[node]['label'] = best_partition.index(cluster)
    # print(graph.nodes('label'))


if __name__ == '__main__':
    main()
