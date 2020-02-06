# In this file I include necessary data and helper functions for genetic algorithm

import numpy as np
import networkx as nx



#### Method to create Initial Population ####
def crt_ip(adjmatrix, ccmat, popsize):
    # Number of nodes
    nd_num = adjmatrix.shape[0]
    # Population matrix
    popmat = []
    
    # For each chromosome of population
    for chrom in range(popsize):
        # Chromosome matrix
        chrommat = np.zeros(shape=(nd_num))
        # For each gene (node) of chromosome
        for i in range(nd_num):
            # Find node's neighbors (indices)
            neighs = np.where(adjmatrix[i] == 1)[0]
            
            if(len(neighs)==0):
                ### SSSOOOOSSS I must define what to do with ISOLATED nodes
                ### Probably I'll put value -1 as neighbor
                chrommat[i] = -1
            else:
                # Find neighbors' cc values
                neigh_cc = ccmat[neighs]
                # Find indices of max neighbors' cc values
                max_cc_neigh = np.where(neigh_cc == neigh_cc.max())[0]
                # If there are more than one neighbor with max value, pick a random one
                if(len(max_cc_neigh)>1):
                    neigh_idx = np.random.choice(max_cc_neigh)
                    chrommat[i] = neighs[neigh_idx]
                else:
                    chrommat[i] = neighs[max_cc_neigh]
        
        # Add chromosome to population matrix
        popmat.append(chrommat)
    
    # Return initial population matrix
    return(popmat)


#### Method to detect clusters in a Chromosome ####
def clusters_from_chromosome(chromosome):
    # Matrix to hold detected clusters
    clusters = []
    # Turn chromosome into node couples
    nodes_c = [[i, int(neigh)] for i, neigh in enumerate(chromosome)]
    # First node as a cluster
    # Check for isolated node
    if(nodes_c[0][1]==-1):
        clusters.append([nodes_c[0][0]])
    else:
        clusters.append(nodes_c[0])
    # For each or the rest node couples
    for n1,n2 in nodes_c[1:]:
        in_cluster = False
        
        clusters_found = []
        new_clusters = []
        
        # If it is an isolated node (neighbor == -1) then create a separate cluster
        if(n2==-1):
            clusters.append([n1])
            continue
        # Check if any one of the nodes exists in an already identified cluster
        for i in range(len(clusters)):
            # If at least one node already belongs to a cluster
            if(n1 in clusters[i] or n2 in clusters[i]):
                # Keep cluster index
                clusters_found.append(i)
                # Put both nodes in the cluster
#                clusters[i].extend([n1,n2])
                in_cluster = True
#                break
            else: # If none node in cluster
                new_clusters.append(clusters[i])

        # If nodes in existing clusters, concatenate them as unified cluster
        if(in_cluster==True):
            conc = [n1,n2]
            for idx in clusters_found:
                conc.extend(clusters[idx])
            new_clusters.append(conc)
        
        else: # If none node in existing cluster, create a new one
            new_clusters.append([n1,n2])
        
        # Update clusters structure
        clusters = new_clusters.copy()
    
    # Remove multiple appearances of nodes in clusters and sort nodes
    clusters = [list(set(cluster)) for cluster in clusters]
    for i in range(len(clusters)):
        clusters[i].sort()
    
    return(clusters)


#### Method to calculate modularity score of a chromosome population ####
def modularity_score_pop(population, graph_object):
    # Matrix to hold population evaluation using Modularity. Each
    # chromosome has its own modularity score
    pop_mod = np.zeros(shape=(len(population)))
    
    # Calculate modularity for each chromosome of population
    for i,chrom in enumerate(population):
        # Find clusters in chromosome
        clusters = clusters_from_chromosome(chrom)
        # Calculate modularity with networkX
        mod = nx.algorithms.community.modularity(graph_object,clusters)
        pop_mod[i] = mod
    
    return(pop_mod)


#### Uniform Cross Over operator on two chromosomes ####
def unif_cross_over_oper(chrom1, chrom2):
    # Number of genes in chromosome
    gen_num = len(chrom1)
    # Create a random binary vector. Each value determines which parent will
    # donate the corresponding gene to the offspring
    rbv = np.random.randint(low=0, high=2, size=gen_num)
    
    # The new offspring
    # Use binary vector's values to populate the genes
    # Apply binary vector on one chromosome and its complement on the other chromosome
    offspring = chrom1*rbv+chrom2*(1-rbv)
    
    # A more detailed way to do exactly the same thing
    # Initially create a zero matrix
#    offspring = np.zeros(shape=gen_num)
#    for i in range(gen_num):
#        if(rbv[i]==1):
#            offspring[i] = chrom1[i]
#        else:
#            offspring[i] = chrom2[i]
    return(offspring)


#### Traditional Mutation operator on a chromosome ####
def trad_mut_oper(chromosome_or, adjmatrix):
    # SOS!!! Use copies otherwise will alter original variables
    chromosome = chromosome_or.copy()
    # Number of genes in chromosome
    gen_num = len(chromosome)
    # Select a random gene of chromosome
    mut_idx = np.random.randint(0,gen_num,1)[0]
    # Variable just for testing reasons
#    or_neigh = chromosome[mut_idx]
    
    # If selected gene is an isolated graph then no mutation occurs
    if(chromosome[mut_idx]==-1):
#        print('Gene %d selected. Isolated gene. No mutation.' %mut_idx)
        return(chromosome)
    # Otherwise find genes neighbors (indices) on original graph using its adjacency matrix
    neighs = np.where(adjmatrix[mut_idx] == 1)[0]
    # If there is only one neigbor then no mutation occurs
    if(len(neighs)==1):
#        print('Gene %d selected. Unique neighbor gene. No mutation.' %mut_idx)
        return(chromosome)
    # Otherwise select a random new neighbor for the gene (no matter the CC value)
    new_neigh = np.random.choice(neighs)
    # New neighbor should be different than the olready existing one
    while(new_neigh == chromosome[mut_idx]):
        new_neigh = np.random.choice(neighs)
    chromosome[mut_idx] = new_neigh
#    print('Gene %d selected. Mutation completed.' %mut_idx)
#    print(np.where(adjmatrix[mut_idx] == 1)[0], or_neigh, new_neigh)
    return(chromosome)


# Method that examines the nodes of a cluster and find the ones that have neighbors
# on original graph outside the cluster
def trans_clusters_nodes(cluster_or, adjmatrix):
    tcn = []
    # For each node (index) in cluster
    for node in cluster_or:
        # Find gene's (node's) neighbors (indices) on original graph
        neighs = list(np.where(adjmatrix[node] == 1)[0])
        # If node's neighbors are all inside the cluster (the neighs set is a
        # subset of the cluster) then procced to next node
        if(set(neighs).issubset(cluster_or)):
            continue
        # Otherwise put node in the trans-clusters nodes list together with its
        # neigbors that are outside the cluster
        # Node's neighbors that don't belong to cluster
        trans_cluster_neighs = np.setdiff1d(np.array(neighs),np.array(cluster_or))
        tcn.append((node,trans_cluster_neighs))
        
    return(tcn)


#### Extended Mutation operator on a chromosome ####
def ext_mut_oper(chromosome_or, adjmatrix):
    # SOS!!! Use copies otherwise will alter original variables
    chromosome = chromosome_or.copy()
    # Find clusters of chromosome
    clusters = clusters_from_chromosome(chromosome)
    # If there is only one cluster, then no mutation occurs
    if(len(clusters)==1):
        return(chromosome)
        
    if(len(clusters)>1):
        # Select a random cluster
        cluster = np.random.choice(np.array(clusters))
        # If cluster correxponds to an isolated node, then no mutation occurs
        if(len(cluster)==1):
            return(chromosome)
        
        # Otherwise must detect the cluster's nodes that have neighbors on original
        # graph that belong to a different cluster
        tr_cl_nds = trans_clusters_nodes(cluster, adjmatrix)
        
        # If there is not such a node, then no mutation occurs
        if(len(tr_cl_nds)==0):
    #        print('No tr_neighs, No mutation')
            return(chromosome)
        
        # Otherwise select a random trans-clusters gene (node) from list. Node are
        # listed together with their trans-clusters neighbors as a tuple
        # Can't random pick directly from a list of tuples so I use indices
        rand_idx = np.random.choice(len(tr_cl_nds))
        gene, tr_cl_neighs = tr_cl_nds[rand_idx]
        # Choose a random of node's trans-clusters neighbors and assign it to the node
    #    or_neigh = chromosome[gene]
        chromosome[gene] = np.random.choice(tr_cl_neighs)
    #    print('gene:%d, or_neigh:%d, new_neigh:%d' %(gene, or_neigh, chromosome[gene]))
    
    return(chromosome)



