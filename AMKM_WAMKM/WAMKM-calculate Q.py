# SNA 2019-2020 Assignment
# Konstantinos Serderidis AEM 46
# AMKM/WAMKM algorithm based on the Weighted adjacent matrix for K-means clustering, Jukai Zhou1,Tong Liu & Jingting Zhu
# https://doi.org/10.1007/s11042-019-08009-x


# version 2: calculate modularity for the set no of clusters
# reference the American football dataset

import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict


# Select the dataset to load
# Different ways to load, directly via netwrokx as G, via .txt files, in .gml format

# Load the reference paper dataset for validation
# Diabetes Load graph as .csv

# df = pd.read_csv(os.getcwd()+"\diabetes.csv")
# X = df.iloc[:, 0:-1].values
# gt = df.iloc[:, -1].values

#Calculate Similarity Matrix W in this case to work based on paper
# W = pairwise_distances(X, metric="euclidean")


# Karate club, as G via networkx

#G = nx.karate_club_graph()
#ground_truth = nx.get_node_attributes(G, 'club')
#gt = [ground_truth [i] for i in G.nodes()]
#gt = np.array([0 if i == 'Mr. Hi' else 1 for i in gt])


# addtional data can be loaded if needed
# dataset = np.loadtxt(os.getcwd()+"\karate_club_unweighted.txt")
# dataset = np.loadtxt(os.getcwd()+"\karate_club_weighted.txt")
# To add edges 
# G.add_weighted_edges_from(dataset)


# American Football Load graph as .txt

G=nx.read_edgelist(os.getcwd()+"\Football.txt", nodetype=int)
ground_truth = np.loadtxt(os.getcwd()+"\GTFootball.txt")
gt= ground_truth[:, -1]


# Emails Load graph as .txt

#G=nx.read_edgelist(os.getcwd()+"\email-Eu-core.txt", nodetype=int)
#ground_truth = np.loadtxt(os.getcwd()+"\email-Eu-core-gt.txt")
#gt= ground_truth[:, -1]


#  Polbooks, Load graph as G to get the ground truth

# G = nx.read_gml(os.getcwd()+"\pollbooks.gml")
# ground_truth = nx.get_node_attributes(G, 'value')
# gt = [ground_truth [i] for i in G.nodes()]
# gt = np.array([0 if i == '0' else 1 if i == '1' else 2 for i in gt])

# Dolphins Load graph as .txt only for Modularity calculation, no ground truth in this case
# G=nx.read_edgelist(os.getcwd()+"\dolphins-edges.txt", nodetype=int)


# Get adjacency-matrix as numpy-array
Adj_Mat = nx.to_numpy_array(G)


# Determine the No of Clusters per dataset
No_clusters = 12


# Initialize tables for metrics based on r for different values of hyperparameter σ (10**(-6)to 10**(r-6)))
r = 2

# Metrics for AMKM
NMI_AMKM = np.arange(1,r, dtype = np.float64)
ACC_AMKM  = np.arange(1,r, dtype = np.float64)
VM_AMKM  = np.arange(1,r, dtype = np.float64)
OM_AMKM = np.arange(1,r, dtype = np.float64)
PUR_AMKM  = np.arange(1,r, dtype = np.float64)
ARS_AMKM  = np.arange(1,r, dtype = np.float64)
AMI_AMKM  = np.arange(1,r, dtype = np.float64)

# Metrics for W-AMKM
NMI_WAMKM = np.arange(1,r, dtype = np.float64)
ACC_WAMKM = np.arange(1,r, dtype = np.float64)
VM_WAMKM  = np.arange(1,r, dtype = np.float64)
OM_WAMKM  = np.arange(1,r, dtype = np.float64)
PUR_WAMKM = np.arange(1,r, dtype = np.float64)
ARS_WAMKM  = np.arange(1,r, dtype = np.float64)
AMI_WAMKM  = np.arange(1,r, dtype = np.float64)

# Metrics for Spectral Clustering
NMI_SPCL = np.arange(1,r, dtype = np.float64)
ACC_SPCL = np.arange(1,r, dtype = np.float64)
VM_SPCL  = np.arange(1,r, dtype = np.float64)
OM_SPCL  = np.arange(1,r, dtype = np.float64)
PUR_SPCL = np.arange(1,r, dtype = np.float64)
ARS_SPCL = np.arange(1,r, dtype = np.float64)
AMI_SPCL = np.arange(1,r, dtype = np.float64)


# Metrics for AMKM PErmutation
PERM_NMI_AMKM = np.arange(1,r, dtype = np.float64)
PERM_ACC_AMKM  = np.arange(1,r, dtype = np.float64)
PERM_VM_AMKM  = np.arange(1,r, dtype = np.float64)
PERM_OM_AMKM = np.arange(1,r, dtype = np.float64)
PERM_PUR_AMKM  = np.arange(1,r, dtype = np.float64)
PERM_ARS_AMKM  = np.arange(1,r, dtype = np.float64)
PERM_AMI_AMKM  = np.arange(1,r, dtype = np.float64)

# Metrics for W-AMKM PErmutation
PERM_NMI_WAMKM = np.arange(1,r, dtype = np.float64)
PERM_ACC_WAMKM = np.arange(1,r, dtype = np.float64)
PERM_VM_WAMKM  = np.arange(1,r, dtype = np.float64)
PERM_OM_WAMKM  = np.arange(1,r, dtype = np.float64)
PERM_PUR_WAMKM = np.arange(1,r, dtype = np.float64)
PERM_ARS_WAMKM = np.arange(1,r, dtype = np.float64)
PERM_AMI_WAMKM  = np.arange(1,r, dtype = np.float64)


# Metrics for Spectral Clustering
PERM_NMI_SPCL = np.arange(1,r, dtype = np.float64)
PERM_ACC_SPCL = np.arange(1,r, dtype = np.float64)
PERM_VM_SPCL  = np.arange(1,r, dtype = np.float64)
PERM_OM_SPCL  = np.arange(1,r, dtype = np.float64)
PERM_PUR_SPCL = np.arange(1,r, dtype = np.float64)
PERM_ARS_SPCL = np.arange(1,r, dtype = np.float64)
PERM_AMI_SPCL = np.arange(1,r, dtype = np.float64)

SIGMA  = np.arange(1,r, dtype = np.float64)


# =============================================================================    
def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)   
# =============================================================================


# Permutation to accomodated for different labeling

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
        permutation.append(new_label)
    return permutation

# source: https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-lists
def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items())



#print metrics

# Run the classification algorithm for different values of σ 
for n in range(1,r):

    # variable values for σ according to r
    σ = 10 ** (n-6)
    γ = 1/(2*σ**2)
    # Adjacent Matrix if W is given
    # Adj_Mat = np.exp(-γ*(W** 2))

    
    # Calculate weights per feature from Adjaceny Mattrix
    D = np.sum(Adj_Mat,axis=0)
    
    # Apply scaling, if needed (not the case)
    # minMaxScaler = MinMaxScaler(feature_range=[0, 1])
    # Adj_Mat = minMaxScaler.fit_transform(Adj_Mat)

    # Calculate sum for all features for W-AMKM
    dsum= np.sum(D,axis=0)

    # Normalize weights and calculate weight vector HNorm
    HNorm = D/dsum

    # Apply Weight Vector to Adjacency Matrix and calculate adjusted Adjazency Matrix Z
    Z= Adj_Mat * HNorm


    # Select Clustering Algorithm K-Means for AMKM
    AMKM = KMeans(n_clusters=No_clusters,n_init=100,random_state=0)
    # =============================================================================
    # Train the model
    AMKM.fit(Adj_Mat)
    # =============================================================================
    

    # Select Clustering Algorithm K-Means for W-AMKM
    WAMKM = KMeans(n_clusters=No_clusters,n_init=100,random_state=0)
    # =============================================================================
    # Train the model, apply directly to Z Matrix
    WAMKM.fit(Z)
    # =============================================================================
    

    # Select Clustering Algorithm Spectral Clustering
    SPCL = SpectralClustering(n_clusters=No_clusters,affinity= 'precomputed', n_init=100, gamma= γ, random_state=0)
    # =============================================================================
       
    # Train the model, apply directly to Adjacency MAtrix
    SPCL.fit(Adj_Mat)
    # =============================================================================

    AMKM_labels = AMKM.labels_
    WAMKM_labels = WAMKM.labels_
    SPCL_labels = SPCL.labels_

    # =============================================================================

    # Compute Metrics per index
    SIGMA[n-1] = σ


    NMI_AMKM[n-1] = normalized_mutual_info_score(gt, AMKM_labels)   
    ACC_AMKM[n-1] = accuracy_score(gt, AMKM_labels)  
    VM_AMKM[n-1]  = v_measure_score(gt, AMKM_labels)    
    OM_AMKM[n-1]  = homogeneity_score(gt, AMKM_labels)
    PUR_AMKM[n-1] = purity_score(gt, AMKM_labels)
    ARS_AMKM[n-1] = adjusted_rand_score(gt, AMKM_labels)
    AMI_AMKM[n-1] = adjusted_mutual_info_score(gt, AMKM_labels)
    
    
    NMI_WAMKM[n-1] = normalized_mutual_info_score(gt, WAMKM_labels)   
    ACC_WAMKM[n-1] = accuracy_score(gt, WAMKM_labels)  
    VM_WAMKM[n-1]  = v_measure_score(gt, WAMKM_labels)    
    OM_WAMKM[n-1]  = homogeneity_score(gt, WAMKM_labels)
    PUR_WAMKM[n-1] = purity_score(gt, WAMKM_labels)
    ARS_WAMKM[n-1] = adjusted_rand_score(gt, WAMKM_labels)
    AMI_WAMKM[n-1] = adjusted_mutual_info_score(gt, WAMKM_labels)
    

    NMI_SPCL[n-1] = normalized_mutual_info_score(gt,SPCL_labels)   
    ACC_SPCL[n-1] = accuracy_score(gt,SPCL_labels)  
    VM_SPCL[n-1]  = v_measure_score(gt, SPCL_labels)    
    OM_SPCL[n-1]  = homogeneity_score(gt, SPCL_labels)   
    PUR_SPCL[n-1] = purity_score(gt, SPCL_labels)
    ARS_SPCL[n-1] = adjusted_rand_score(gt, WAMKM_labels)
    AMI_SPCL[n-1] = adjusted_mutual_info_score(gt, WAMKM_labels)

    
    permutation_AMKM = find_permutation(No_clusters, gt, AMKM_labels)
    #print(permutation_AMKM)
    permutation_WAMKM = find_permutation(No_clusters, gt, WAMKM_labels)
    #print(permutation_WAMKM)
    permutation_SPCL = find_permutation(No_clusters, gt, SPCL_labels)
    #print(permutation_SPCL)

    new_AMKM_labels=np.zeros_like(gt)
    new_WAMKM_labels=np.zeros_like(gt)
    new_SPCL_labels=np.zeros_like(gt)

    new_AMKM_labels = [ permutation_AMKM[label] for label in AMKM.labels_]   # permute the labels
    new_AMKM_labels = np.asarray(new_AMKM_labels)
    #print("AMKM Accuracy score is", accuracy_score(gt, new_AMKM_labels))

    new_WAMKM_labels = [ permutation_WAMKM[label] for label in WAMKM.labels_]   # permute the labels
    new_WAMKM_labels = np.asarray(new_WAMKM_labels)
    #print("WAMKM Accuracy score is", accuracy_score(gt, new_WAMKM_labels))

    new_SPCL_labels = [ permutation_SPCL[label] for label in SPCL.labels_]   # permute the labels
    new_SPCL_labels = np.asarray(new_SPCL_labels)
    #print("SPCL_Accuracy score is", accuracy_score(gt, new_SPCL_labels))

    # Compute Metrics per index  with permutation
    
    PERM_NMI_AMKM[n-1] = normalized_mutual_info_score(gt, new_AMKM_labels)   
    PERM_ACC_AMKM[n-1] = accuracy_score(gt, new_AMKM_labels)  
    PERM_VM_AMKM[n-1]  = v_measure_score(gt, new_AMKM_labels)    
    PERM_OM_AMKM[n-1]  = homogeneity_score(gt, new_AMKM_labels)
    PERM_PUR_AMKM[n-1] = purity_score(gt, new_AMKM_labels)
    PERM_ARS_AMKM[n-1] = adjusted_rand_score(gt, new_AMKM_labels)
    PERM_AMI_AMKM[n-1] = adjusted_mutual_info_score(gt, new_AMKM_labels)
    
    PERM_NMI_WAMKM[n-1] = normalized_mutual_info_score(gt, new_WAMKM_labels)   
    PERM_ACC_WAMKM[n-1] = accuracy_score(gt, new_WAMKM_labels)  
    PERM_VM_WAMKM[n-1]  = v_measure_score(gt, new_WAMKM_labels)    
    PERM_OM_WAMKM[n-1]  = homogeneity_score(gt, new_WAMKM_labels)
    PERM_PUR_WAMKM[n-1] = purity_score(gt, new_WAMKM_labels)
    PERM_ARS_WAMKM[n-1] = adjusted_rand_score(gt, new_WAMKM_labels)
    PERM_AMI_WAMKM[n-1] = adjusted_mutual_info_score(gt, new_WAMKM_labels)

    PERM_NMI_SPCL[n-1] = normalized_mutual_info_score(gt, new_SPCL_labels)   
    PERM_ACC_SPCL[n-1] = accuracy_score(gt, new_SPCL_labels)  
    PERM_VM_SPCL[n-1]  = v_measure_score(gt, new_SPCL_labels)    
    PERM_OM_SPCL[n-1]  = homogeneity_score(gt, new_SPCL_labels)
    PERM_PUR_SPCL[n-1] = purity_score(gt, new_SPCL_labels)
    PERM_ARS_SPCL[n-1] = adjusted_rand_score(gt, new_SPCL_labels)
    PERM_AMI_SPCL[n-1] = adjusted_mutual_info_score(gt, new_SPCL_labels)
    

#print(ACC_AMKM)
#print(ACC_WAMKM)
#print(ACC_SC)

#print(NMI_AMKM)
#print(NMI_WAMKM)
#print(NMI_SC)
    
# Plots for assessment

#plt.figure()
#plt.semilogx(SIGMA, ACC_AMKM, 'k', label='AMKM')
#plt.semilogx(SIGMA, ACC_WAMKM, 'b',  label='WAMKM')
#plt.semilogx(SIGMA, ACC_SPCL, 'g',  label='SPECTRAL CLUSTERING')

#plt.title("WAMKM algorith assessment")
#plt.xlabel('The sigma (σ) value 10(x)')
#plt.ylim(0.35, 0.7)
#plt.xlim(10**(-5), 10**14)
#plt.ylabel('ACC')

#plt.legend()
# Display 'ticks' in x-axis and y-axis
# =============================================================================
#plt.xticks()
#plt.yticks()
# =============================================================================

# Show plot
# =============================================================================
#plt.show()
# =============================================================================



# retrieve the communities partitions with reference node No AMKM/WAMKM
C =sorted(list_duplicates(WAMKM_labels))
# C =sorted(list_duplicates(AMKM_labels))

# extract only the communities partitions
B=[(C[i][1]) for i in range(0,No_clusters)]

# re-arrange / shift nodes no if indexing starts from 1 (for football dataset since numbering starts from 0)
#for i in range(0,No_clusters): 
#    B[i]= [x+1 for x in B[i]]

# for polbooks dataset
#G =nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

# calculate Modularity Q
Q = nx.algorithms.community.modularity(G, B)
           
