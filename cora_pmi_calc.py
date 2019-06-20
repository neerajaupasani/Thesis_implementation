#import statements
import random
from collections import defaultdict
from itertools import combinations,chain
from scipy.sparse import coo_matrix
import pickle
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt



# defining class with random-walk generator


#def convert(tup, di):
    #for a, b in tup:
        #di.setdefault(a, []).append(b)
    #return di


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
    with open('cora-co-occurrence.txt', 'w+') as f:
        for item in walks:
            f.write("%s\n" % item)
    #return walks
    #p1=pd.DataFrame(walks)
    #p1_dict=p1.to_dict('list')
    #print(p1_dict)
    #new_list=Counter(p1_dict)
    #print(new_list)

    node_occur = (pd.get_dummies(pd.DataFrame(walks), prefix='', prefix_sep='').groupby(level=0, axis=1).sum())



    cooccur_matrix = node_occur.T.dot(node_occur)
    cooccur_matrix.values[(np.c_[:len(cooccur_matrix)],) * 2] = 0
    print("The co-occurrences are \n",cooccur_matrix)
    co_dense=cooccur_matrix.values
    print("Converting the df to a numpy array\n",co_dense)
    global co_coo
    co_coo=coo_matrix(co_dense)
    print("Converting to a sparse coordinate matrix\n",co_coo)

    #global mat
    #mat=cooccur_matrix.values
    #print("Converting the co-occurrence df to numpy array:",mat)

random_walk(10,40)

def pmi(pmi_threshold=None):
    global co_coo
    row_sum = np.squeeze(np.asarray(np.sum(co_coo, axis=1)))
    total_sum = np.sum(row_sum)
    row_prob = row_sum / total_sum
    col_sum = np.squeeze(np.asarray(np.sum(co_coo, axis=0)))
    col_prob=col_sum / total_sum
    pmi_matrix = np.log2(co_coo.data / (total_sum * row_prob[co_coo.row] * col_prob[co_coo.col]))

    if pmi_threshold is None:
        pmi_rowcol = (co_coo.row, co_coo.col)
    else:
        thresholded_mat = (pmi_matrix >= pmi_threshold)
        pmi_matrix = pmi_matrix[thresholded_mat]
        pmi_rowcol = (co_coo.row[thresholded_mat], co_coo.col[thresholded_mat])
    return coo_matrix((pmi_matrix, pmi_rowcol), co_coo.shape)

pmi_values=pmi(pmi_threshold=-2.5)
print("PMI matrix\n",pmi_values)

def generate_embeddings():
    global co_coo
    global mat_float
    mat_float=co_coo.asfptype()
    u, _, _ = svds(mat_float,k=32)
    print("Generating node embeddings and writing them to file\n")
    with open('cora-embeddings.txt', 'w') as f:
        for value in u:
            f.write("%s\n" % value)

generate_embeddings()

def plot_pmi():
    global mat_float
    #float_array = mat_float.toarray()
    pmi_embedded = TruncatedSVD(n_components=2).fit_transform(mat_float)
    x_val = pmi_embedded[:, 0]
    y_val = pmi_embedded[:, 1]
    color = ("red", "green", "blue")
    plt.scatter(x_val, y_val, c=color)
    plt.show()

plot_values = plot_pmi()
print("The plot of PMI matrix\n", plot_values)


