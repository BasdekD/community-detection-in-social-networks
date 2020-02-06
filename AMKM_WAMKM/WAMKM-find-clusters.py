# SNA 2019-2020 Assignment
# Konstantinos Serderidis AEM 46
# AMKM/WAMKM algorithm based on the Weighted adjacent matrix for K-means clustering, Jukai Zhou1,Tong Liu & Jingting Zhu
# https://doi.org/10.1007/s11042-019-08009-x

# version 1: find the most suitable number of clusters per dataset

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


# Get adjacency-matrix as numpy-array
Adj_Mat = nx.to_numpy_array(G)


# Initialize tables for metrics based on r for different No of Clusters r
No_Clusters= 20


r = No_Clusters

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


#metrics for error calculation for the Elbow Method
SSE_AMKM=[]
SSE_WAMKM =[]

#print metrics

# Run the classification algorithm for different values of σ 
for n in range(1,No_Clusters):
  
 
    # Calculate weights per feature from Adjacency Mattrix
    D = np.sum(Adj_Mat,axis=0)
    
    # Apply scaling, if needed (not the case)
    # minMaxScaler = MinMaxScaler(feature_range=[0, 1])
    # Adj_Mat = minMaxScaler.fit_transform(Adj_Mat)

    # Calculate sum for all features for W-AMKM
    dsum= np.sum(D,axis=0)

    # Normalize weights and calculate weight vector HNorm
    HNorm = D/dsum

    # Apply Weight Vector to Adjacency Matrix and calculate Z Matrix
    Z= Adj_Mat * HNorm


    # Select Clustering Algorithm K-Means for AMKM
    AMKM = KMeans(n_clusters=n,n_init=100,random_state=0)
    # =============================================================================
    # Train the model
    AMKM.fit(Adj_Mat)
    # =============================================================================
    SSE_AMKM.append( AMKM.inertia_)
    

    # Select Clustering Algorithm K-Means for W-AMKM
    WAMKM = KMeans(n_clusters=n,n_init=100,random_state=0)
    # =============================================================================
    # Train the model, apply directly to Z Matrix
    WAMKM.fit(Z)
    # =============================================================================
    SSE_WAMKM.append( WAMKM.inertia_)

  
    AMKM_labels = AMKM.labels_
    WAMKM_labels = WAMKM.labels_

    
    # =============================================================================

    # Compute Metrics per index

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
    
    
    permutation_AMKM = find_permutation(n, gt, AMKM_labels)
    #print(permutation_AMKM)
    permutation_WAMKM = find_permutation(n, gt, WAMKM_labels)
    #print(permutation_WAMKM)


    new_AMKM_labels=np.zeros_like(gt)
    new_WAMKM_labels=np.zeros_like(gt)

    new_AMKM_labels = [ permutation_AMKM[label] for label in AMKM.labels_]   # permute the labels
    new_AMKM_labels = np.asarray(new_AMKM_labels)
    #print("AMKM Accuracy score is", accuracy_score(gt, new_AMKM_labels))

    new_WAMKM_labels = [ permutation_WAMKM[label] for label in WAMKM.labels_]   # permute the labels
    new_WAMKM_labels = np.asarray(new_WAMKM_labels)
    #print("WAMKM Accuracy score is", accuracy_score(gt, new_WAMKM_labels))

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
    

#print(ACC_AMKM)
#print(ACC_WAMKM)

#print(NMI_AMKM)
#print(NMI_WAMKM)

#print(PUR_AMKM)
#print(PUR_WAMKM)


# Plot stored results
# =============================================================================
# ADD COMMANDS TO PLOT RESULTS HERE
# =============================================================================


plt.figure(1)

plt.plot(range(1, No_Clusters), PERM_AMI_WAMKM,'k+', label='Adjusted Mutual Info Score')
plt.plot(range(1, No_Clusters), PERM_ARS_WAMKM,'r--', label='Adjusted Rand Score')
plt.plot(range(1, No_Clusters), PERM_VM_WAMKM,'bs', label='V Measure Score')
plt.plot(range(1, No_Clusters), PERM_NMI_WAMKM,'g^', label='Normalized Mutual Info Score')
plt.plot(range(1, No_Clusters), PERM_ACC_WAMKM,'y', label='Accuracy')

plt.xlim(1, No_Clusters)
#plt.ylim(0, 0.5)
plt.title('Metrics - WAMKM American Football dataset')
plt.xlabel('No of clusters')
plt.ylabel('Metrics')

plt.legend()
plt.show()

plt.figure(2)

# plt.plot(range(1, r), SSE_AMKM)
# plt.title('Elbow method - AMKM')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()


plt.plot(range(1, r), SSE_WAMKM)
plt.title('Elbow method - WAMKM American Football dataset')
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show()

