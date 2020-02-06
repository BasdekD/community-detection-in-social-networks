# Automate analysis of multiple datasets
import numpy as np
import pandas as pd
import os
from sys import stdout
import networkx as nx

# Import methods necessary for analysis
from graph_modules import calc_cc_mat,crt_adj_mat
from ga_modules import (crt_ip,modularity_score_pop,unif_cross_over_oper,trad_mut_oper,
                        ext_mut_oper,clusters_from_chromosome)


#  Local path to datasets
dsp = 'Datasets'

# Available datasets
dsnames = [
#        'dolphins-(159_edges).txt',
#        'email-Eu-core-(25571_edges).txt',
#        'karate-(78_edges).txt',
#        'football-(613_edges).txt',
#        'polbooks-(441_edges).txt',
        'ref_paper_graph-(19_edges).txt'
        ]


# Graph and GA parameters
# Population size
pop_size = 200
# Percent of population to participate in crossover operation. Select the best chromosomes
p_c = 0.1
# Crossover rate. The possibility for a selected chromosome couple to be crossovered
co_r = 0.2
# Mutation rate
mut_r = 0.15
# Mutation extention rate
mut_e_r = 0.02
# Maximum number of iterations wo any improvement
r = 50
# Max number of generations (iterations)
iters = 500

# How many times to repeat analysis
repeats = 1

# Total Best Chromosome and the corresponding modularity score
best_chrom = 0
best_mod = 0

# Current population to be analyzed and its modularity scores
pop = 0
pop_mod = 0


# Method to evaluate a population of chromosomes, based on their modularity score
# Returns the chromosome and the corresponding modularity scores matrices sorted
# in descending order, based on modularity score.
def eval_pop(population, graph_object):
    # Calculate Modularity score of current population's chromosomes
    popul_mod = modularity_score_pop(population, graph_object)
    # Sort modularity matrix in descending order. Returns the corresponding indices.
    desc_idxs = np.argsort(popul_mod)[::-1]
    # Use indices to sort both population and modularity score matrices
    pop_mod_srt = popul_mod[desc_idxs]
    pop_srt = [population[i] for i in desc_idxs]
    
    # Return sorted matrices
    return(pop_srt, pop_mod_srt)


# Method to randomly select a pair of chromosomes using probabilities based on
# their modularity scores
def roulette_wheel_selection(mod_mat, pair_num):
    pairs = []
    # Turning modularity score into probabilities
    probs =[mod/mod_mat.sum() for mod in mod_mat]
    for i in range(pair_num):
        # Select a random couple (indices), using calculated probabilities
        pair = np.random.choice(len(mod_mat),2,probs)
        # If pair contains same elements, repeat
        while(pair[0] == pair[1]):
            pair = np.random.choice(len(mod_mat),2,probs)
        pairs.append(pair)
        
    return(pairs)


# CrossOver operation
def crossover_op(population,popmod,prc_pop,co_rate):
    # The new population after cross over
    offspring = []
    
    # Prepare the chromosome pairs to crossover
    pairs_num = int(np.ceil(len(population)/2))
    # Use ONLY the first prc_pop chromosomes of population for crossover
    chroms_to_co = int(len(population)*prc_pop)
    pairs_idxs = roulette_wheel_selection(popmod[:chroms_to_co], pairs_num)

    # For each pair of chromosomes
    for pair in pairs_idxs:
        # Pick a random number from 0 to 1
        co = np.random.uniform(0,1)
        # If random number is up to crossover rate
        if(co <= co_rate):
            # then perform crossover operation twice to get two new offsprings
            offs1 = unif_cross_over_oper(population[pair[0]], population[pair[1]])
            offspring.append(offs1)
            offs2 = unif_cross_over_oper(population[pair[0]], population[pair[1]])
            offspring.append(offs2)
        # Otherwise the two parents will pass in the new population
        else:
            offspring.extend([population[pair[0]], population[pair[1]]])
    
    # Return only the proper number of new chromosomes
    return(offspring[:len(population)])

    
# Traditional mutation operation
def trad_mutation_op(population,adjmat,mut_rate):
    print("Traditional Mutation operator:")
    # The new population after mutation
    offspring = []
    mut_counts = 0
    # For each chromosome
    for chrom in population:
        # Pick a random number from 0 to 1
        mut = np.random.uniform(0,1)
        # If random number is up to mutation rate
        if(mut <= mut_rate):
            # then perform traditional mutation operation to get a mutated offspring
            offs = trad_mut_oper(chrom, adjmat)
            offspring.append(offs)
            mut_counts += 1
            stdout.write("\r\t%d chromosome(s) mutated!!!" % mut_counts)
            stdout.flush()
        # Otherwise the unaltered chromosome will pass in the new population
        else:
            offspring.append(chrom)
    stdout.write("\n")
    return(offspring)


# Extended mutation operation
def extended_mutation_op(population,adjmat,mut_e_r):
    print("Extended Mutation operator:")
    # The new population after mutation
    offspring = []
    mut_counts = 0
    # For each chromosome
    for chrom in population:
        # Pick a random number from 0 to 1
        e_mut = np.random.uniform(0,1)
        # If random number is up to mutation rate
        if(e_mut <= mut_e_r):
            # then perform traditional mutation operation to get a mutated offspring
            offs = ext_mut_oper(chrom, adjmat)
            offspring.append(offs)
            mut_counts += 1
            stdout.write("\r\t%d chromosome(s) mutated!!!" % mut_counts)
            stdout.flush()
        # Otherwise the unaltered chromosome will pass in the new population
        else:
            offspring.append(chrom)
    stdout.write("\n")
    return(offspring)


def reset_numbering_to_zero(edges_list):
    # Find smallest node numbering in dataset
    snm = edges_list.min()
    # If smallest node numbering > 0 then subtract this number from all nodes labels
    if(snm > 0):
        edges_list = edges_list - snm
    return(edges_list, snm)




########################  Main Algorithm  ####################################

# For each dataset or a selected dataset
for dsname in dsnames:
    print('@'*20 + '   Start analysing %s dataset  ' %dsname[:-4] + '@'*20 + '\n' )
    
    # Read dataset (txt file)
    # Networks are represented as couples of nodes corresponding to graph edges
    edges = np.loadtxt(os.path.join(dsp,dsname), delimiter=" ", skiprows=0)
    
    # For convenience all nodes numbering should start from 0. If not then I alter
    # the numbering accordingly
    edges, num_alt = reset_numbering_to_zero(edges)
    
    # Create a networkX graph object
    egraph = nx.Graph()
    # Construct the Graph from edges
    egraph.add_edges_from(edges)
    
    # Number of nodes in network. I assume that it should be equal to maximum node
    # numbering occuring in dataset + 1.
    nodes = int(edges.max() + 1)
    
    # Calculate matrix to check values
    adj_mat = crt_adj_mat(nodes, edges)
    
    # List to hold results from each repetition
    rep_res = []
    
    # Repetitions of analysis
    for repeat in range(repeats):
        print('#'*15 + '   Start repeat %d (of %d in total)  ' %((repeat+1), repeats) + '#'*15 + '\n' )
        # Calculate Clustering Coefficient matrix for all nodes in graph, using original
        # graph's adjacency matrix
        cc_mat = calc_cc_mat(adj_mat)
        
        # Create Initial Population
        ip = crt_ip(adj_mat, cc_mat, pop_size)
        
        # Evaluate initial population. Returns chromosome and modularity matrices sorted
        # in descending order, based on modularity
        pop, pop_mod = eval_pop(ip,egraph)
        
        # First chromosome is the best chromosome, based on modularity score
        best_chrom = pop[0]
        best_mod = pop_mod[0]
        
        ########## Start genetic operations  ##############
        
        # Number of generations analyzed
        iter_num = 0
        # Number of generations wo population change
        pops_wo_change = 0
        
        while(pops_wo_change<r and iter_num<iters):
            print('-'*25 + '   Generation %d   ' %iter_num + '-'*25)
            # Flag to indicate whether the population changed during a generation
            pop_changed = False
        
            # Crossover operation
            new_gen = crossover_op(pop,pop_mod,p_c,co_r)
            # Evaluate new generation
            new_gen_ev, new_gen_mod_ev = eval_pop(new_gen,egraph)
            # Compare old and new genaration (best modularity score)
            # If new generation improves best modularity score vs old generation
            if(new_gen_mod_ev[0] > best_mod):
                # Then set new generation as the current generation of analysis
                pop = new_gen_ev.copy()
                pop_mod = new_gen_mod_ev.copy()
                # Update best chromosome data
                best_chrom = new_gen_ev[0]
                best_mod = new_gen_mod_ev[0]
                # Update flag
                pop_changed = True
            
            
            # Traditional mutation operation
            new_gen = trad_mutation_op(pop, adj_mat,mut_r)
            # Evaluate new generation
            new_gen_ev, new_gen_mod_ev = eval_pop(new_gen,egraph)
            # Compare old and new genaration (best modularity score)
            # If new generation improves best modularity score vs old generation
            if(new_gen_mod_ev[0] > best_mod):
                # Then set new generation as the current generation of analysis
                pop = new_gen_ev.copy()
                pop_mod = new_gen_mod_ev.copy()
                # Update best chromosome data
                best_chrom = new_gen_ev[0]
                best_mod = new_gen_mod_ev[0]
                # Update flag
                pop_changed = True
            
            # Extended mutation operation
            new_gen = extended_mutation_op(pop, adj_mat,mut_r)
            # Evaluate new generation
            new_gen_ev, new_gen_mod_ev = eval_pop(new_gen,egraph)
            # Compare old and new genaration (best modularity score)
            # If new generation improves best modularity score vs old generation
            if(new_gen_mod_ev[0] > best_mod):
                # Then set new generation as the current generation of analysis
                pop = new_gen_ev.copy()
                pop_mod = new_gen_mod_ev.copy()
                # Update best chromosome data
                best_chrom = new_gen_ev[0]
                best_mod = new_gen_mod_ev[0]
                # Update flag
                pop_changed = True
                
            # Update counters
            iter_num += 1
            if(pop_changed):
                pops_wo_change = 0
            else:
                pops_wo_change += 1
        
        # Save repetition's results
        clusters = clusters_from_chromosome(best_chrom)
        clusters_num = len(clusters)
        # Reset the initial node numbering
        if(num_alt):
            clusters = [list(np.array(i)+1) for i in clusters]
        # Fill the result structure for this repetition
        rep_res.append([dsname[:-4],pop_size,iter_num,co_r,mut_r,best_mod,clusters_num,clusters,best_chrom])

        
        # When analysis is completed
        print('\n\nGenetic Algorithm results:')
        print('\tFinished after %d generations' %iter_num)
        print('\tBest chromosome modularity: %f' %best_mod)
        print('\tClusters found: %d' %clusters_num)
        print('\tClusters: ', clusters)

    # Save dataset's total results
    # Dataframe to hold dataset's total results
    headers = [
            'Dataset',
            'Population',
            'Iterations',
            'COrate',
            'MUTrate',
            'Best mod',
            'Clusters#',
            'Clusters',
            'Best chromosome'
            ]
    ds_df = pd.DataFrame(rep_res, columns = headers)



######### Choose whether to locally save the dataset's results  ################
if(1):
    # Local folder to save dataset's analysis results
    ptdsaf = 'Datasets analysis results'
    # If folder doesn't exists, create it
    if not os.path.exists(ptdsaf):
        os.makedirs(ptdsaf)
    # The file name to be used, with useful algorithm's parameters
    fname = dsname[:-4] + ' (pop: %d, Generations: %d, COrate: %.2f, MUTrate: %.4f)' %(pop_size,iters,co_r,mut_r)
    # Store dataframe file as a pickle
    ds_df.to_pickle(os.path.join(ptdsaf, fname))



################ Code to read stored results  ############################
if(0):
    # Local folder to hold saved dataset's results
    ptdsaf = 'Datasets analysis results'
    
    # Important parameters to find desired file
    # Choose the dataset
    dsnames = [
            'dolphins-(159_edges).txt',
            'email-Eu-core-(25571_edges).txt',
            'karate-(78_edges).txt',
            'football-(613_edges).txt',
            'polbooks-(441_edges).txt',
            'ref_paper_graph-(19_edges).txt'
            ]
    dsname = dsnames[2][:-4]
    
    # Give GA parameters values used to produce the results
    # Population size
    pop_size = 200
    # Crossover rate. The possibility for a selected chromosome couple to be crossovered
    co_r = 0.2
    # Mutation rate
    mut_r = 0.15
    # Max number of generations (iterations)
    iters = 50
    
    # Filename based on parameters above
    # All files found in storage folder
    dts_stored = [f for f in os.listdir(ptdsaf) if os.path.isfile(os.path.join(ptdsaf, f))]
    # Choose the right file
    dts_fname = [f for f in dts_stored if (dsname in f and str(pop_size) in f and str(co_r) in f and str(mut_r) in f and str(iters) in f)][0]

    # Read stored results dataframe
    dts = pd.read_pickle(os.path.join(ptdsaf,dts_fname))


