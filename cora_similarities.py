#import statements
import random
import csv
from collections import defaultdict
from itertools import combinations,chain
from scipy.sparse import csr_matrix
import pickle
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import nltk
import networkx as nx
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt



# doing random walk over the graph

def random_walk(no_of_walks,walk_length):

    p = "cora-graph.txt"
    result = []
    with open(p, "r") as fp:
        for i in fp.readlines():
            tmp = i.split(" ")
            try:
                result.append((int(tmp[0]), int(tmp[1])))
            #result.append((eval(tmp[0]), eval(tmp[1])))
            except:pass
    d=defaultdict(list)
    for v, k in result:
        d[v].append(k)

    walks=list()
    for node in list(d.keys()):
        for i in range(no_of_walks):
            the_walk = [node]
            walk_list = list(d[node])
            if len(walk_list) == 0:
                break
            next_node = random.choice(walk_list)
            the_walk.append(next_node)
            for k in range(walk_length):
                walk_list = list(d[the_walk[-1]])
                if len(walk_list) == 0:
                    break
                context_node = random.choice(walk_list)
                the_walk.append(context_node)
            walks.append(the_walk)
    walks = [list(map(str, walk)) for walk in walks]
    with open('cora_random_walk.txt', 'w+') as f:
        for item in walks:
            f.write("%s\n" % item)

    # converting the list of walks to python dictionary

    array_dict = {t[0]: t[1:] for t in walks}
    # converting the dictionary to an adjacency matrix

    get_values = {k: [v.strip() for v in vs] for k, vs in array_dict.items()}

    adj_edges = [(a, b) for a, bs in get_values.items() for b in bs]

    adj_df = pd.DataFrame(adj_edges)

    adj_matrix = pd.crosstab(adj_df[0], adj_df[1])

    adj_matrix.to_csv(r'cora_adjacency_graph.txt', header=None, index=None, sep=',')

    global adj_mat
    adj_mat = adj_matrix.as_matrix()



random_walk(5,20)

def cosine_matrix(cosine_threshold=None):
    global adj_mat
    global cosine_matrix
    cosine_matrix=cosine_similarity(adj_mat)
    if cosine_threshold is not None:
        thresholded_mat=cosine_matrix > cosine_threshold
        cosine_matrix[thresholded_mat] = 0
    else:
        print("No Threshold applied")

    return cosine_matrix


cosine_values=cosine_matrix(cosine_threshold=0.6)
print("The cosine matrix \n",cosine_values)

def jaccard_matrix(jaccard_threshold=None):
    global adj_mat
    #converting the integer array into a CSR matrix
    csr_array=csr_matrix(adj_mat)
    #binarizing the csr matrix
    #csr_array[csr_array >= 200] = 0
    #csr_array[csr_array != 0] = 1
    bin_array = csr_array.toarray()

    intersection_jac = ((csr_array * bin_array.T) + (bin_array * csr_array.T))

    row_sum = np.sum(csr_array, axis=1)
    union_jac = np.repeat(row_sum, intersection_jac.shape[0], axis=1) + np.repeat(row_sum.T, intersection_jac.shape[0], axis=0)









    global jac_matrix
    jac_matrix = intersection_jac / union_jac
    if jaccard_threshold is not None:
        jac_threshold = jac_matrix > jaccard_threshold
        jac_matrix[jac_threshold] = 0
    else:
        print("No Threshold applied")



    return jac_matrix




jac_values=jaccard_matrix(jaccard_threshold=0.6)
print("The jaccard matrix is\n",jac_values)

def leicht_holme_newman_matrix(leicht_threshold = None):
    global adj_mat
    # converting the integer array into a CSR matrix
    lh1_array = csr_matrix(adj_mat)
    bin1_array = lh1_array.toarray()

    intersection1 = ((lh1_array * bin1_array.T) + (bin1_array * lh1_array.T))

    # to add the elements from lh1 matrix
    lh1_array_sum = lh1_array.sum(axis=1)

    # to add the elements from bin1_matrix
    bin1_array_sum = bin1_array.sum(axis=1)

    # Hadamard product of lh1_array_sum and bin1_array_sum
    hp_mat = np.multiply(lh1_array_sum, bin1_array_sum)


    global lh1_matrix
    lh1_matrix = intersection1 / hp_mat
    if leicht_threshold is not None:
        lh1_threshold = lh1_matrix > leicht_threshold
        lh1_matrix[lh1_threshold] = 0
    else:
        print("No Threshold applied")

    return lh1_matrix

lh1_values = leicht_holme_newman_matrix(leicht_threshold=0.5)
print("The Leicht-Holme-Newman matrix is\n", lh1_values)

def generate_cosine_embeddings():
    global cosine_matrix
    #mat_float=cosine_matrix.asfptype()
    u, _, _ = svds(cosine_matrix,k=32)
    print("Generating node embeddings from cosine matrix and writing them to file\n")
    with open('cora_cosine_embeddings.txt', 'w+') as f:
        for value in u:
            f.write("%s\n" % value)

generate_cosine_embeddings()

def plot_cosine():
    global cosine_matrix
    #cosine_array = cosine_matrix.toarray()
    cosine_embedded = TruncatedSVD(n_components=2).fit_transform(cosine_matrix)
    x_val = cosine_embedded[:, 0]
    y_val = cosine_embedded[:, 1]
    color = ("red", "green", "blue")
    plt.scatter(x_val, y_val, c=color)
    plt.show()

plot_values = plot_cosine()
print("The plot of cosine matrix\n", plot_values)

def generate_jaccard_embeddings():
    global jac_matrix
    #mat_float=cosine_matrix.asfptype()
    u, _, _ = svds(jac_matrix,k=32)
    print("Generating node embeddings from jaccard matrix and writing them to file\n")
    with open('cora_jaccard_embeddings.txt', 'w+') as f:
        for value in u:
            f.write("%s\n" % value)

generate_jaccard_embeddings()

def plot_jaccard():
    global jac_matrix
    #cosine_array = cosine_matrix.toarray()
    jac_embedded = TruncatedSVD(n_components=2).fit_transform(jac_matrix)
    x_val = jac_embedded[:, 0]
    y_val = jac_embedded[:, 1]
    color = ("red", "green", "blue")
    plt.scatter(x_val, y_val, c=color)
    plt.show()

plot_values = plot_jaccard()
print("The plot of jaccard matrix\n", plot_values)

def generate_lh1_embeddings():
    global lh1_matrix
    #mat_float=cosine_matrix.asfptype()
    u, _, _ = svds(lh1_matrix,k=32)
    print("Generating node embeddings from lh1 matrix and writing them to file\n")
    with open('cora_leicht_embeddings.txt', 'w+') as f:
        for value in u:
            f.write("%s\n" % value)

generate_lh1_embeddings()

def plot_lh1():
    global lh1_matrix
    #cosine_array = cosine_matrix.toarray()
    lh1_embedded = TruncatedSVD(n_components=2).fit_transform(lh1_matrix)
    x_val = lh1_embedded[:, 0]
    y_val = lh1_embedded[:, 1]
    color = ("red", "green", "blue")
    plt.scatter(x_val, y_val, c=color)
    plt.show()

plot_values = plot_lh1()
print("The plot of lh1 matrix\n", plot_values)


