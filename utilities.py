from itertools import combinations, product
import operator
import networkx as nx


def loadDummyDataset():
    """
    A function to load a small dataset used for development and testing purposes
    """
    init_nodes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    init_edges = [('0', '8'), ('0', '1'), ('0', '4'),
                  ('1', '0'), ('1', '8'), ('1', '4'),
                  ('2', '10'), ('2', '9'),
                  ('3', '6'),
                  ('4', '7'), ('4', '1'), ('4', '0'),
                  ('5', '9'), ('5', '8'), ('5', '6'), ('5', '7'),
                  ('6', '5'), ('6', '8'), ('6', '3'), ('6', '7'),
                  ('7', '6'), ('7', '5'), ('7', '4'),
                  ('8', '9'), ('8', '5'), ('8', '6'), ('8', '1'), ('8', '0'),
                  ('9', '2'), ('9', '5'),
                  ('10', '2')]

    graph = nx.Graph()
    graph.add_nodes_from(init_nodes)
    graph.add_edges_from(init_edges)
    return graph


def loadDataset(dataset):
    """
    A function to load datasets saved locally
    """
    graph = nx.read_edgelist(
        "C:\\Users\\Dimitris.DESKTOP-VO6DORB\\Desktop\\ΑΠΘ\\SNA\\Project_Implementation\\datasets\\"+dataset)
    return graph


# A FUNCTION WHICH CALCULATES THE LOCAL CLUSTERING COEFFICIENT OF A GIVEN NODE
def getLocalClusteringCoefficient(G, n):
    """
    param G: NetworkX graph
    param n: A node of G
    returns: Local clustering coefficient of n
    """
    degree_of_node = G.degree[n]
    # A PARAM FOR THE NUMBER OF CONNECTIONS BETWEEN NEIGHBORS OF NODE n
    num_of_neighbors_con = 0
    edges = []
    # GETTING ALL NEIGHBORS OF n
    neighborhood = G.subgraph(G.neighbors(n))
    for (u, v) in neighborhood.edges:
        # WE WANT ONLY THE CONNECTION OF n's NEIGHBORS NOT n's CONNECTIONS
        if n not in (u, v) and (u, v) not in edges and (v, u) not in edges and u in neighborhood\
                and v in neighborhood and u != v:
            edges.append((u, v))
            num_of_neighbors_con += 1
    # BY DEFINITION AS DESCRIBED IN THE PAPER
    if degree_of_node == 0 or degree_of_node == 1:
        return 0
    else:
        # THE FORMULA OF LOCAL CLUSTERING COEFFICIENT AS DESCRIBED IN THE PAPER
        CCn = (2*num_of_neighbors_con)/(degree_of_node*(degree_of_node - 1))
        return CCn


# A FUNCTION CALCULATING SIMILARITY INDEX OF TWO ADJACENT NODES
def similarityIndex(G, u, v):
    """
    param G: A networkx graph
    param u: A node of G
    param v: A node of G which is also a neighobor of u
    returns: The similarity index of u and v
    """
    com_neighbors = nx.common_neighbors(G, u, v)
    similarity = 0
    for n in com_neighbors:
        similarity += getLocalClusteringCoefficient(G, n)
    return similarity


# A FUNCTION TO RETURN ALL POSSIBLE VALID COMBINATION OF A GIVEN LIST OF TUPLES
def getLegitCombinations(most_similar_node_list):
    """
    param most_similar_node_list: A list of pairs of the most similar nodes
    returns: All legit combinations of edges
    """
    different_values = []
    # WE EXTRACT ALL DIFFERENT VALUES OF FIRST NODES IN AN EDGE AND STORE THEM IN A LIST
    for (u, v) in most_similar_node_list:
        if u not in different_values:
            different_values.append(u)
    # WE TAKE ALL THE POSSIBLE COMBINATIONS WITH LENGTH EQUAL TO THE DIFFERENT VALUES OF NODES
    all_combinations = list(combinations(most_similar_node_list, len(different_values)))

    legit_combinations_list = []
    # WE TRAVERSE ALL POSSIBLE COMBINATIONS AND KEEP IN A LIST ONLY THE VALID ONES
    for tp in all_combinations:
        legit_combination = True
        for i in range(0, len(different_values)):
            # BECAUSE OUR all_combinations LIST IS SORTED THE VALID COMBINATIONS WILL HAVE THE DIFFERENT NODE VALUES
            # IN ORDER AND EXACTLY ONE TIME
            if tp[i][0] != different_values[i]:
                legit_combination = False
        if legit_combination:
            legit_combinations_list.append(list(tp))
            if len(legit_combinations_list) == 100:
                return legit_combinations_list
    print(len(legit_combinations_list))
    return list(legit_combinations_list)

# def getLegitCombinations(most_similar_node_list):
#     legit_combinations = []
#     i = 0
#     while i < len(most_similar_node_list) - 1:
#         temp_comb = []
#         counter = 0
#         if not legit_combinations:
#             legit_combinations.append([most_similar_node_list[i]])
#             continue
#         if most_similar_node_list[i][0] == most_similar_node_list[i+1][0]:
#             counter = 1
#             while most_similar_node_list[i][0] == most_similar_node_list[i+counter][0]:
#                 counter += 1
#                 temp_comb.append([most_similar_node_list[counter + i - 1]])
#             legit_combinations = list(product(legit_combinations, temp_comb))
#         else:
#             for comb in legit_combinations:
#                 comb.append(most_similar_node_list[i])
#                 i += 1
#                 continue
#         i += counter



def getUniqueConnectedComponents(connected_components):
    """
    param connected_components: The legit combinations of edges
    returns: Unique connected components after removing duplicates
    """
    unique_connected_components = []
    for component in connected_components:
        for (u, v) in component:
            for (w, x) in component:
                if (u, v) == (x, w) or (v, u) == (w, x):
                    component.remove((w, x))
        if component not in unique_connected_components:
            unique_connected_components.append(component)

    return unique_connected_components


def getBasicCommunities(component):
    """
    param component: A unique component of edges
    returns: A basic community structure
    """
    basic_communities = []
    for (u, v) in component:
        community_found = False
        if not basic_communities:
            basic_communities.append([(u, v)])
        else:
            curr_num_of_communities = len(basic_communities)
            for i in range(0, curr_num_of_communities):
                curr_num_of_edges = len(basic_communities[i])
                for j in range(0, curr_num_of_edges):
                    if u in basic_communities[i][j] or v in basic_communities[i][j]:
                        basic_communities[i].append((u, v))
                        community_found = True
                        break
            if not community_found:
                basic_communities.append([(u, v)])

    return mergeBasicCommunities(basic_communities)


def mergeBasicCommunities(basic_communities):
    """
    A function called by getBasicCommunities function to optimize primitive community structure by adding for example
    [(2,1 0)] to [(5, 3), (7, 9), (2, 10)]
    """
    for community in basic_communities:
        merged = False
        for pair in community:
            for other_community in basic_communities:
                if pair in other_community and other_community != community:
                    new_community = list(set(community + other_community))
                    basic_communities.append(new_community)
                    basic_communities.remove(community)
                    basic_communities.remove(other_community)
                    merged = True
                    break
            if merged:
                break
        if merged:
            mergeBasicCommunities(basic_communities)
            break

    return basic_communities


def mergingStrategy(G, basic_community, threshold):
    """
    A function implementing a merging strategy that merges small communities to larger ones if they have fewer nodes
    than a given threshold
    """
    if threshold > len(G.nodes):
        print("Threshold can't be greater than the number of nodes in the graph. Setting threshold to " +
              (str(int(len(G.nodes) / 2))))
        threshold = int(len(G.nodes) / 2)
    for component in basic_community:
        comp_graph = nx.Graph()
        comp_graph.add_edges_from(component)
        degrees = {}
        merged = False
        if len(list(comp_graph.nodes)) < threshold:
            max_CC = 0
            for n in comp_graph:
                node_CC = getLocalClusteringCoefficient(G, n)
                if max_CC <= node_CC and not not(G.neighbors(n) - comp_graph.nodes):
                    max_CC = node_CC
                    node_with_max_CC = n
            neighbors = G.neighbors(node_with_max_CC)
            for node in neighbors:
                if node not in comp_graph.nodes:
                    degrees[node] = G.degree[node]
            neighbor_with_max_degree = max(degrees.items(), key=operator.itemgetter(1))[0]
            for other_component in basic_community:
                for edge in other_component:
                    if neighbor_with_max_degree in edge:
                        new_component = list(set(component + other_component))
                        basic_community.append(new_component)
                        basic_community.remove(component)
                        basic_community.remove(other_component)
                        merged = True
                        break
                if merged:
                    mergingStrategy(G, basic_community, threshold)
                    break
    return basic_community
