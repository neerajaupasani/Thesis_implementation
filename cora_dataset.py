#import statements

import networkx as nx
import numpy as np
import math
import pandas as pd
from itertools import combinations,chain
import nltk
from collections import Counter
from scipy.sparse import coo_matrix
from numpy import count_nonzero
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict




#defining class

class Cora:
    def loading_file(self):
        graph = "citeseer-edgelist.txt"
        type_of_graph = nx.Graph()

        # to read from the file
        global new_graph
        new_graph = nx.read_edgelist(graph, create_using=type_of_graph, nodetype=int)




        # printing the information of number of nodes and edges in the graph

        print(nx.info(new_graph))

        # to check and remove self loops if any
        for u, v in list(new_graph.edges()):
            if u == v:
                new_graph.remove_edge(u, v)
        print("After checking for self-loops", new_graph.number_of_edges())

        # to generate a random sequence of nodes

    def random_walk(self, path_length, start=None):
        global new_graph
        if start:
            global path
            path = [start]
        else:
            path = [np.random.choice(list(new_graph.nodes()))]
        while len(path) < path_length:
            cur = path[-1]
            if len(list(new_graph.neighbors(cur))) > 0:
                path.append(np.random.choice(list(new_graph.neighbors(cur))))
            else:
                path.append(path[0])

        return path

    # generating an artificial corpus of walks from the paths

    def artificial_corpus(self, no_walks, walk_len):
        global walks
        walks = []
        global new_graph

        for i in list(new_graph.nodes()):
            for j in range(no_walks):
                walks.append(self.random_walk(walk_len, start=i))

        return walks

    # to print the walks

    def print_walks(self):
        global walks
        with open('wiki_corpus.txt', 'w+') as f:
            for item in walks:
                f.write(str(item))

    # function to calculate the cosine similarity between nodes

    def cosine(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]


        common_uv = len(list(set(neighbor_u) & set(neighbor_v)))

        if common_uv == 0:
            cos_uv = 0

        else:
            cos_uv = common_uv / math.sqrt(len(neighbor_u) * len(neighbor_v))

        return cos_uv

    # function to calculate the jaccard similarity between nodes

    def jaccard(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]


        common_uv = len(list(set(neighbor_u) & set(neighbor_v)))
        union_uv = len(list(set(neighbor_u) | set(neighbor_v)))

        if common_uv == 0:
            jac_uv = 0

        else:
            jac_uv = common_uv / union_uv

        return jac_uv

    # function to calculate the lhn1 similarity between nodes

    def lhn1(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]


        common_uv = len(list(set(neighbor_u) & set(neighbor_v)))

        if common_uv == 0:
            lhn1_uv = 0

        else:
            lhn1_uv = common_uv / (len(neighbor_u) * len(neighbor_v))

        return lhn1_uv

    # function to calculate the sorensen similarity between nodes

    def sorensen(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]

        common_uv = 2 * len(list(set(neighbor_u) & set(neighbor_v)))

        if common_uv == 0:
            sorensen_uv = 0

        else:
            sorensen_uv = common_uv / (len(neighbor_u) + len(neighbor_v))

        return sorensen_uv

    # function to calculate the HPI similarity between nodes

    def HPI(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]


        common_uv = len(list(set(neighbor_u) & set(neighbor_v)))

        if common_uv == 0:
            hpi_uv = 0

        else:
            hpi_uv = common_uv / min(len(neighbor_u), len(neighbor_v))

        return hpi_uv

    # function to calculate the HDI similarity between nodes

    def HDI(self,u,v):
        global new_graph
        neighbor_u = [n for n in new_graph.neighbors(u)]
        neighbor_v = [n for n in new_graph.neighbors(v)]

        common_uv = len(list(set(neighbor_u) & set(neighbor_v)))

        if common_uv == 0:
            hdi_uv = 0

        else:
            hdi_uv = common_uv / max(len(neighbor_u), len(neighbor_v))

        return hdi_uv

    # sliding window on the corpus to generate pairs and calculate the similarity measures for the pairs

    def node_pairs(self, window):


        global flatten, pairs, walks, u, v, windows


        pairs = []
        flatten = lambda list: [item for sublist in list for item in sublist]
        windows = flatten([list(nltk.ngrams(c, window * 2 + 1)) for c in walks])




        for win in windows:



            for i in range(window * 2 + 1):

                u = win[window]
                v = win[i]
                cosine_uv = self.cosine(u,v)
                jaccard_uv = self.jaccard(u,v)
                lhn1_uv = self.lhn1(u,v)
                sorensen_uv = self.sorensen(u,v)
                hpi_uv = self.HPI(u,v)
                hdi_uv = self.HDI(u,v)


                pairs.append((u, v, cosine_uv, jaccard_uv, lhn1_uv, sorensen_uv, hpi_uv, hdi_uv))





        return pairs

    # counting the occurrence of generated pairs (u,v) in the corpus.

    def co_occurrence_mat(self):
        global walks, u, v, new_graph

        no = new_graph.order()

        global cxy
        cxy = Counter()

        for walk in walks:
            for u, v in map(sorted, combinations(walk, 2)):
                cxy[(u, v)] += 1

        # generating the co-occurrence matrix

        k = np.array(list(cxy.keys()))
        val = np.array(list(cxy.values()))

        table = np.zeros((no,no), dtype=int)

        table[k[:, 0], k[:, 1]] = val

        table[k[:, 1], k[:, 0]] = val


        np.fill_diagonal(table, 0)

        mat = pd.DataFrame(table)
        #mat.fillna(0, inplace=True)
        #mat.to_csv(r'coo_matrix.csv')

        coo_array = mat.values

        check_sparsity = 1.0 - (count_nonzero(coo_array) / float(coo_array.size))
        print(check_sparsity)

        global co_occur
        co_occur = coo_matrix(coo_array)

    # building a thresholded PMI matrix from the co-occurrence matrix generated.

    def pmi_mat(self, pmi_threshold = None):
        global co_occur
        #co_occur.astype(float)
        row_sum = np.squeeze(np.asarray(np.sum(co_occur, axis=1)))
        total_sum = np.sum(row_sum)
        row_prob = row_sum / total_sum
        col_sum = np.squeeze(np.asarray(np.sum(co_occur, axis=0)))
        col_prob = col_sum / total_sum
        # global pmi_matrix
        pmi_matrix = np.log2(co_occur.data / (total_sum * row_prob[co_occur.row] * col_prob[co_occur.col]))


        if pmi_threshold is None:
            pmi_rowcol = (co_occur.row, co_occur.col)
        else:
            thresholded_mat = (pmi_matrix <= pmi_threshold)
            pmi_matrix = pmi_matrix[thresholded_mat]
            pmi_rowcol = (co_occur.row[thresholded_mat], co_occur.col[thresholded_mat])

        return coo_matrix((pmi_matrix, pmi_rowcol), co_occur.shape)

    # performing SVD on the thresholded PMI matrix to capture the final node embeddings.

    # Use of TruncatedSVD for efficiently handling sparse matrices.

    def temp_pmi(self, output):
        pmi = self.pmi_mat(-3.0)
        U = TruncatedSVD(n_components=128, algorithm="arpack")

        svd = U.fit_transform(pmi)
        print("Generating embeddings with pmi score")

        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    # saving the pairs with their similarity measures in the file

    def print_pairs(self):
        global pairs
        global pairs_df


        pairs_df = pd.DataFrame(pairs, columns=['node1','node2','cosine_score','jaccard_score','lhn1_score','sorensen_score','hpi_score','hdi_score'])


        remove_dupl = pairs_df.drop_duplicates(keep='first')


        remove_dupl.to_csv(r'wiki_statistics.csv', index = False)

    # Building the thresholded cosine similarity matrix.

    def cosine_mat(self, cosine_threshold = None):
        global new_graph
        global u,v

        no = new_graph.order()

        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.cosine(u,v))
            rowlist.append(collist)

        cos_mat = np.array(rowlist)

        if cosine_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = cos_mat <= cosine_threshold
            cos_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(cos_mat) / float(cos_mat.size))
        print(check_sparsity)

        cos_df = pd.DataFrame(cos_mat)
        #cos_df.to_csv(r'cos_matrix.csv')

        cos_array = cos_df.values

        global cosine
        cosine = coo_matrix(cos_array)


    # Building the thresholded jaccard similarity matrix.



    def jaccard_mat(self, jaccard_threshold = None):
        global new_graph
        global u, v

        no = new_graph.order()

        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.jaccard(u, v))
            rowlist.append(collist)

        jac_mat = np.array(rowlist)
        if jaccard_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = jac_mat <= jaccard_threshold
            jac_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(jac_mat) / float(jac_mat.size))
        print(check_sparsity)

        jac_df = pd.DataFrame(jac_mat)
        #jac_df.to_csv(r'jac_matrix.csv')

        jac_array = jac_df.values

        global jaccard
        jaccard = coo_matrix(jac_array)


    # Building the thresholded lhn1 similarity matrix.


    def lhn1_mat(self, lhn1_threshold = None):
        global new_graph
        global u, v

        no = new_graph.order()
        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.lhn1(u, v))
            rowlist.append(collist)

        lhn1_mat = np.array(rowlist)
        if lhn1_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = lhn1_mat <= lhn1_threshold
            lhn1_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(lhn1_mat) / float(lhn1_mat.size))
        print(check_sparsity)

        lhn1_df = pd.DataFrame(lhn1_mat)
        #lhn1_df.to_csv(r'lhn1_matrix.csv')

        lhn1_array = lhn1_df.values

        global lhn1
        lhn1 = coo_matrix(lhn1_array)

    # Building the thresholded sorensen similarity matrix.

    def sorensen_mat(self, so_threshold=None):
        global new_graph
        global u, v

        no = new_graph.order()

        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.sorensen(u, v))
            rowlist.append(collist)

        so_mat = np.array(rowlist)

        if so_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = so_mat <= so_threshold
            so_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(so_mat) / float(so_mat.size))
        print(check_sparsity)

        so_df = pd.DataFrame(so_mat)
        #so_df.to_csv(r'sor_matrix.csv')

        so_array = so_df.values

        global sorensen
        sorensen = coo_matrix(so_array)

    # Building the thresholded HPI similarity matrix.

    def hpi_mat(self, hpi_threshold=None):
        global new_graph
        global u, v

        no = new_graph.order()

        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.HPI(u, v))
            rowlist.append(collist)

        hpi_mat = np.array(rowlist)

        if hpi_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = hpi_mat <= hpi_threshold
            hpi_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(hpi_mat) / float(hpi_mat.size))
        print(check_sparsity)

        hpi_df = pd.DataFrame(hpi_mat)
        #hpi_df.to_csv(r'hpi_matrix.csv')

        hpi_array = hpi_df.values

        global hpi
        hpi = coo_matrix(hpi_array)

    # Building the thresholded HDI similarity matrix.

    def hdi_mat(self, hdi_threshold=None):
        global new_graph
        global u, v

        no = new_graph.order()

        rowlist = []
        for u in range(no):
            collist = []
            for v in range(no):
                collist.append(self.HDI(u, v))
            rowlist.append(collist)

        hdi_mat = np.array(rowlist)

        if hdi_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = hdi_mat <= hdi_threshold
            hdi_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(hdi_mat) / float(hdi_mat.size))
        print(check_sparsity)

        hdi_df = pd.DataFrame(hdi_mat)
        #hdi_df.to_csv(r'hdi_matrix.csv')

        hdi_array = hdi_df.values

        global hdi
        hdi = coo_matrix(hdi_array)


    # performing SVD on the thresholded similarity matrices to capture the final node embeddings.


    def temp_cosine(self, output):
        global cosine
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(cosine)
        print("Generating embeddings with cosine score")
        #print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    def temp_jaccard(self, output):
        global jaccard
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(jaccard)
        print("Generating embeddings with jaccard score")
        # print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    def temp_lhn1(self, output):
        global lhn1
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(lhn1)
        print("Generating embeddings with lhn1 score")
        # print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    def temp_sor(self, output):
        global sorensen
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(sorensen)
        print("Generating embeddings with sorensen score")
        # print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    def temp_hpi(self, output):
        global hpi
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(hpi)
        print("Generating embeddings with hpi score")
        # print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')

    def temp_hdi(self, output):
        global hdi
        U = TruncatedSVD(n_components=128, algorithm="arpack")
        svd = U.fit_transform(hdi)
        print("Generating embeddings with hdi score")
        # print(type(svd))
        pd.DataFrame(svd).to_csv(output, header=None, index=False, float_format='%8.7f', sep=',')














obj = Cora()
obj.loading_file()
obj.artificial_corpus(10,40)
obj.print_walks()
obj.node_pairs(5)
obj.co_occurrence_mat()
obj.temp_pmi('cite_pembed.csv')
obj.print_pairs()
obj.cosine_mat(0.0)
obj.jaccard_mat(0.0)
obj.lhn1_mat(0.0)
obj.sorensen_mat(0.0)
obj.hpi_mat(0.0)
obj.hdi_mat(0.0)
obj.temp_cosine('cora_cembed.csv')
obj.temp_jaccard('cora_jembed.csv')
obj.temp_lhn1('cora_lembed.csv')
obj.temp_sor('cora_sembed.csv')
obj.temp_hpi('cora_hembed.csv')
obj.temp_hdi('cora_dembed.csv')










