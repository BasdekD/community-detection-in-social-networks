import networkx as nx
import math
import numpy as np
import os

def calcstructuralsimilarity(graph):

    for e in graph.edges:
        structural_similarity = (len(sorted(nx.common_neighbors(graph, e[0], e[1]))) + 2) / \
                                math.sqrt(
                                    (len(sorted(graph.neighbors(e[0]))) + 1) * (len(sorted(graph.neighbors(e[1]))) + 1))
        graph[e[0]][e[1]]['Structural Similarity'] = structural_similarity

def findcommunities(graph, e, m):

    # Find Core Nodes

    core = []
    for n in graph.nodes:
        count = 0
        for e1, e2 in graph.edges(n):
            if graph[e1][e2]['Structural Similarity'] >= e:
                count += 1
        if count >= m:
            core.append(n)
    # print(core)

    # Define Clusters

    for node in graph.nodes:
        graph.nodes[node]['mark'] = 0
    label = 1
    for node in core:
        if graph.nodes[node]['mark'] == 1:
            label_old = graph.nodes[node]['label']
            for neighbor in nx.neighbors(graph, node):
                if graph.nodes[neighbor]['mark'] == 0:
                    if graph[node][neighbor]['Structural Similarity'] >= e:
                        graph.nodes[neighbor]['mark'] = 1
                        graph.nodes[neighbor]['label'] = label_old
        else:
            graph.nodes[node]['mark'] = 1
            graph.nodes[node]['label'] = label
            for neighbor in nx.neighbors(graph, node):
                if graph.nodes[neighbor]['mark'] == 0:
                    if graph[node][neighbor]['Structural Similarity'] >= e:
                        graph.nodes[neighbor]['mark'] = 1
                        graph.nodes[neighbor]['label'] = label
            label += 1
    # print(graph.nodes('label'))

    # Find bridges and outliers

    for node in graph.nodes:
        if graph.nodes[node]['mark'] == 0:
            labels = []
            for neighbor in nx.neighbors(graph, node):
                if graph.nodes[neighbor]['mark'] == 1:
                    labels.append(graph.nodes[neighbor]['label'])
            if len(set(labels)) >= 2:
                graph.nodes[node]['label'] = -1  # 'Bridge'
            else:
                graph.nodes[node]['label'] = -2  # 'Outlier'
    # print(graph.nodes('label'))


    # Evaluation with modularity score

    partition = [[] for x in range(len(graph.nodes)+1)]
    for node in graph.nodes:
        if not((graph.nodes[node]['label'] == -1) | (graph.nodes[node]['label'] == -2)):
            partition[graph.nodes[node]['label']].append(node)
    no_of_communities = sum(len(x) > 0 for x in partition)
    n = no_of_communities
    n += 1
    for node in graph.nodes:
        if (graph.nodes[node]['label'] == -1) | (graph.nodes[node]['label'] == -2):
            partition[n].append(node)
            n += 1
    partition = [x for x in partition if x != []]
    modularity = nx.algorithms.community.modularity(graph, partition)
    return modularity, partition, no_of_communities

def testevaluation (datasetname):
    if datasetname == "test_graph.txt":
       m = 3
    else:
        m = 2
    e = 0.7
    graph = nx.read_edgelist(os.getcwd() + '\\' + datasetname)
    calcstructuralsimilarity(graph)
    modularity, partition, no_of_communities = findcommunities(graph, e, m)
    print(partition, no_of_communities)
    print(graph.nodes('label'))
    nx.write_gexf(graph, "file2.gexf")


# Testing in 3 different datasets "test_graph.txt", "CG1.txt", "CG2.txt"

#testevaluation("CG1.txt")




# Import graph from file

#graph = nx.read_edgelist(os.getcwd()+"\karate-edges.txt")
#graph = nx.read_edgelist(os.getcwd()+"\dolphins-edges.txt")
#graph = nx.read_edgelist(os.getcwd()+"\email-Eu-core.txt")
graph = nx.read_edgelist(os.getcwd()+"\out.dimacs10-football.txt")
#graph = nx.read_edgelist(os.getcwd()+"\polbooks-edges.txt")


# Calculate Structural Similarity

calcstructuralsimilarity(graph)

# Find Best Community Structure

best_e = 0
best_m = 0
max_modularity = 0
for e in np.arange(0.1, 1.0, 0.05):
    for m in range(1, 10):
        modularity, partition, no_of_communities = findcommunities(graph, e, m)
        print("e:", e, " m:", m, " Modularity Score:", modularity)
        if max_modularity < modularity:
            max_modularity = modularity
            best_e = e
            best_m = m
modularity, partition, no_of_communities = findcommunities(graph, best_e, best_m)
print("Best modularity and parameters", "\ne:", best_e, " m:", best_m, " Modularity Score:",
      max_modularity, " Number of communitites", no_of_communities,
      "\nGraph Labels:", graph.nodes('label'), "\nGraph Partitions:", partition)


# Create file to import in Gephi
nx.write_gexf(graph, "file2.gexf")



