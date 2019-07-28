#import statements

import networkx as nx
import numpy as np
import math
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations, chain
from sklearn import preprocessing
from scipy.sparse.linalg import svds
import nltk
from collections import OrderedDict,defaultdict
from scipy.sparse import coo_matrix
from numpy import count_nonzero




#defining class

class Cora:
    def loading_file(self):
        graph = "cora-edgelist.txt"
        type_of_graph = nx.Graph()

        # to read from the file
        global new_graph
        new_graph = nx.read_edgelist(graph, create_using=type_of_graph, nodetype=int)

        global node_list
        node_list = list(new_graph.nodes())


        node_df = pd.DataFrame({'vertices': node_list})
        node_df.to_csv(r'cora_node_vocab.csv',index=False)




        print(nx.info(new_graph))

        # to check and remove self loops if any
        for u, v in new_graph.edges():
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
        with open('cora_corpus.txt', 'w+') as f:
            for item in walks:
                f.write(str(item))

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

    #def count_co_occurrence(self, u, v):
        #global walks
        #counter = []
        #for l in walks:
            #for u in l:
                #for v in l:
                    #counter[u][v] += 1







    def node_pairs(self, window):

        global walks
        global pairs
        global u,v,cosine_uv,jaccard_uv,lhn1_uv,new_graph,windows


        pairs = []
        flatten = lambda list: [item for sublist in list for item in sublist]
        windows = flatten([list(nltk.ngrams(c, window * 2 + 1)) for c in walks])



        #num_window = 0

        for win in windows:

            #print(num_window, 'window', win)

            for i in range(window * 2 + 1):
                u = win[window]
                v = win[i]
                cosine_uv = self.cosine(u,v) ## define the other measures (jaccard, LHN)  and counter for co-occurance pairs
                jaccard_uv = self.jaccard(u,v)
                lhn1_uv = self.lhn1(u,v)

                pairs.append((u, v, cosine_uv, jaccard_uv, lhn1_uv))






            #num_window += 1

        return pairs



    def print_pairs(self):
        global pairs
        global pairs_df


        pairs_df = pd.DataFrame(pairs, columns=['node1','node2','cosine_score','jaccard_score','lhn1_score'])
        #pairs_df.to_csv(r'cora_statistics.csv', index=False)




        #pairs_df['co_occurrence_count'] = pairs_df.groupby(['node1', 'node2'])['node1'].transform('size')

        remove_dupl = pairs_df.drop_duplicates(keep='first')

        remove_dupl.to_csv(r'cora_statistics.csv', index = False)

        #row_mat = pairs_df['node1'].tolist()
        #col_mat = pairs_df['node2'].tolist()
        #pairs_list = pd.DataFrame(np.column_stack([row_mat, col_mat]), columns=['node1', 'node2'])
        #pairs_list.to_csv(r'node_pairs.csv')

    def cosine_mat(self, cosine_threshold = None):
        global new_graph
        global pairs,u,v
        rowlist = []
        for u in list(new_graph.nodes()):
            collist = []
            for v in list(new_graph.nodes()):
                collist.append(self.cosine(u,v))
            rowlist.append(collist)

        cos_mat = np.array(rowlist)

        if cosine_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = cos_mat >= cosine_threshold
            cos_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(cos_mat) / float(cos_mat.size))
        print(check_sparsity)

        cos_df = pd.DataFrame(cos_mat)
        cos_df.to_csv(r'cos_matrix.csv')

        cos_array = cos_df.values

        global cosine
        cosine = coo_matrix(cos_array)





    def jaccard_mat(self, jaccard_threshold = None):
        global new_graph
        global pairs, u, v
        rowlist = []
        for u in list(new_graph.nodes()):
            collist = []
            for v in list(new_graph.nodes()):
                collist.append(self.jaccard(u, v))
            rowlist.append(collist)

        jac_mat = np.array(rowlist)
        if jaccard_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = jac_mat >= jaccard_threshold
            jac_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(jac_mat) / float(jac_mat.size))
        print(check_sparsity)

        jac_df = pd.DataFrame(jac_mat)
        jac_df.to_csv(r'jac_matrix.csv')

        jac_array = jac_df.values

        global jaccard
        jaccard = coo_matrix(jac_array)




    def lhn1_mat(self, lhn1_threshold = None):
        global new_graph
        global pairs, u, v
        rowlist = []
        for u in list(new_graph.nodes()):
            collist = []
            for v in list(new_graph.nodes()):
                collist.append(self.lhn1(u, v))
            rowlist.append(collist)

        lhn1_mat = np.array(rowlist)
        if lhn1_threshold is None:
            print("No Threshold applied")

        else:
            thresholded_mat = lhn1_mat >= lhn1_threshold
            lhn1_mat[thresholded_mat] = 0

        check_sparsity = 1.0 - (count_nonzero(lhn1_mat) / float(lhn1_mat.size))
        print(check_sparsity)

        lhn1_df = pd.DataFrame(lhn1_mat)
        lhn1_df.to_csv(r'lhn1_matrix.csv')

        lhn1_array = lhn1_df.values

        global lhn1
        lhn1 = coo_matrix(lhn1_array)

    def co_occurrence_mat(self):
        global node_list, walks

        co_occurrences = OrderedDict((node, OrderedDict((node, 0) for node in node_list)) for node in node_list)

        # Find the co-occurrences:
        for l in walks:
            for i in range(len(l)):
                for item in l[:i] + l[i + 1:]:
                    co_occurrences[l[i]][item] += 1

        coo_df = pd.DataFrame(co_occurrences, columns=co_occurrences.keys(),index=co_occurrences.keys())

        # setting the diagonal to 0
        for i in range(len(coo_df.index)):
            for j in range(len(coo_df.columns)):
                if i == j:
                    coo_df.loc[i, j] = 0


        coo_df.to_csv(r'coo_matrix.csv')

        coo_array = coo_df.values

        check_sparsity = 1.0 - (count_nonzero(coo_array) / float(coo_array.size))
        print(check_sparsity)

        global co_occur
        co_occur = coo_matrix(coo_array)



    def pmi_mat(self, pmi_threshold = None):
        global co_occur
        row_sum = np.squeeze(np.asarray(np.sum(co_occur, axis=1)))
        total_sum = np.sum(row_sum)
        row_prob = row_sum / total_sum
        col_sum = np.squeeze(np.asarray(np.sum(co_occur, axis=0)))
        col_prob = col_sum / total_sum
        # global pmi_matrix
        pmi_matrix = np.log2(co_occur.data / (total_sum * row_prob[co_occur.row] * col_prob[co_occur.col]))
        print(type(pmi_matrix))

        if pmi_threshold is None:
            pmi_rowcol = (co_occur.row, co_occur.col)
        else:
            thresholded_mat = (pmi_matrix <= pmi_threshold)
            pmi_matrix = pmi_matrix[thresholded_mat]
            pmi_rowcol = (co_occur.row[thresholded_mat], co_occur.col[thresholded_mat])

        return coo_matrix((pmi_matrix, pmi_rowcol), co_occur.shape)

    def run_svd_pmi(self, input, dimension, embed_out):
        global co_occur
        cora_df = pd.read_csv(input)

        mat_float = co_occur.astype(float)

        u, _, _ = svds(mat_float, dimension)
        print("Generating embeddings with pmi score")

        pd.DataFrame(u).to_csv(embed_out, header=None, index=False, float_format='%8.7f', sep=',')

        #pmi_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None,
                                                                                             #index=False,
                                                                                             #float_format='%8.7f',
                                                                                             #sep=' ')

    def run_svd_cosine(self, input, dimension, embed_out):
        global cosine, new_graph
        cora_df = pd.read_csv(input)

        mat_float = cosine.astype(float)



        u, _, _ = svds(mat_float, dimension)
        print("Generating embeddings with cosine score")





        pd.DataFrame(u).to_csv(embed_out, header=None, index=False, float_format='%8.7f', sep=',')

        #cosine_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None,
                                                                                             #index=False,
                                                                                             #float_format='%8.7f',
                                                                                             #sep=' ')
        #global embeddings

        #embeddings = {}
        #for i in list(new_graph.nodes()):
            #embeddings[i] = u[i]





    def run_svd_jaccard(self, input, dimension, embed_out):
        global jaccard
        cora_df = pd.read_csv(input)

        mat_float = jaccard.astype(float)

        u, _, _ = svds(mat_float, dimension)
        print("Generating embeddings with jaccard score")

        pd.DataFrame(u).to_csv(embed_out, header=None, index=False, float_format='%8.7f', sep=',')

        #jaccard_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None,
                                                                                             #index=False,
                                                                                             #float_format='%8.7f',
                                                                                             #sep=' ')

    def run_svd_lhn1(self, input, dimension, embed_out):
        global lhn1
        cora_df = pd.read_csv(input)

        mat_float = lhn1.astype(float)

        u, _, _ = svds(mat_float, dimension)
        print("Generating embeddings with lhn1 score")

        pd.DataFrame(u).to_csv(embed_out, header=None, index=False, float_format='%8.7f', sep=',')

        #lhn1_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None,
                                                                                             #index=False,
                                                                                             #float_format='%8.7f',
                                                                                             #sep=' ')







        #coo_df = pd.DataFrame(coo_mat)
        #coo_df.to_csv(r'coo_matrix.csv')



        # Print the matrix:
        #print(' ', ' '.join(co_occurrences.keys()))
        #for name, values in co_occurrences.items():
            #print(name, ' '.join(str(i) for i in values.values()))










    #def print_mat(self):
        #global cos_mat


        #new_df = pd.DataFrame(pairs_df)

    #def co_occur_mat(self):

        #pmi_df = pd.read_csv('cora_statistics.csv', index_col=0)

        #new_df = pd.DataFrame(pmi_df)

        #new_df.rename(columns={'node1': 'n1', 'node2': 'n2', 'cosine_score': 'c', 'jaccard_score': 'j', 'lhn1_score': 'l', 'co_occurrence_count': 'count'},
                      #inplace=True)

        #pmi_pivot = pmi_df.drop_duplicates(keep = 'first')

        #pmi_pivot = pmi_df.pivot(index='node2', columns='node1', values='cosine_score')

        #cora_coocc = remove_dupl.pivot('node2', 'node1')
        #val = pmi_pivot.fillna(0)



        #cora_coocc.to_csv(r'cora_pmi.csv')
        #val = cora_coocc.fillna(0)

        #pmi_mat = pd.DataFrame(val)
        #pmi_mat.values[[np.arange(pmi_mat.shape[0])] * 2] = 0
        #pmi_mat.to_csv(r'cora_pmi.csv')











obj = Cora()
obj.loading_file()
obj.artificial_corpus(5,40)
obj.print_walks()
obj.node_pairs(5)
obj.print_pairs()
obj.cosine_mat(0.5)
obj.jaccard_mat(0.5)
obj.lhn1_mat(0.5)
obj.co_occurrence_mat()
obj.pmi_mat(-2.5)
obj.run_svd_pmi('cora_node_vocab.csv',128,'cora_pembed.csv')
obj.run_svd_cosine('cora_node_vocab.csv',128,'cora_cembed.csv')
obj.run_svd_jaccard('cora_node_vocab.csv',128,'cora_jembed.csv')
obj.run_svd_lhn1('cora_node_vocab.csv',128,'cora_lembed.csv')









