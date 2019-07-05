#import statements

import networkx as nx
import numpy as np
import math
import pandas as pd
from scipy.sparse import coo_matrix
from itertools import combinations, chain
from sklearn import preprocessing
from scipy.sparse.linalg import svds




#defining class
class Wiki:
    def loading_file(self):
        graph = "wiki-edgelist.txt"
        type_of_graph = nx.Graph()

        # to read from the file
        global new_graph
        new_graph = nx.read_edgelist(graph, create_using=type_of_graph, nodetype=int)

        # to generate a random sequence of nodes

    def random_walk(self, path_length, start=None):
        #global new_graph
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

        with open('wiki_paths.txt', 'w+') as f:
            for item in path:
                f.write("%i\n" % item)

    def cosine(self):
        #global path

        f = open('wiki_cosine.txt', "w+")
        for u, v in zip(path, path[1:]):
            neighbor_u = [n for n in new_graph.neighbors(u)]
            neighbor_v = [n for n in new_graph.neighbors(v)]

            common_uv = len(list(set(neighbor_u) & set(neighbor_v)))
            if common_uv == 0:
                cos_uv = 0
            else:
                cos_uv = common_uv / (math.sqrt(len(neighbor_u) * len(neighbor_v)))
            #print(cos_uv)


            f.write("%i % i %.2f\n" % (u, v, cos_uv))

    def jaccard(self):
        #global path
        f = open('wiki_jaccard.txt', "w+")
        for u, v in zip(path, path[1:]):
            neighbor_u = [n for n in new_graph.neighbors(u)]
            neighbor_v = [n for n in new_graph.neighbors(v)]

            common_uv = len(list(set(neighbor_u) & set(neighbor_v)))
            if common_uv == 0:
                jac_uv = 0
            else:
                jac_uv = common_uv / (len(neighbor_u) | len(neighbor_v))
            # print(cos_uv)
            f.write("%.2f\n" % jac_uv)

    def lhn1(self):
        #global path
        f = open('wiki_lhn1.txt', "w+")
        for u, v in zip(path, path[1:]):
            neighbor_u = [n for n in new_graph.neighbors(u)]
            neighbor_v = [n for n in new_graph.neighbors(v)]

            common_uv = len(list(set(neighbor_u) & set(neighbor_v)))
            if common_uv == 0:
                lhn1_uv = 0
            else:
                lhn1_uv = common_uv / (len(neighbor_u) * len(neighbor_v))
            # print(cos_uv)
            f.write("%.2f\n" % lhn1_uv)

    # to combine the similarity scores in one file

    def similarity_scores(self):
        files = ['wiki_cosine.txt', 'wiki_jaccard.txt', 'wiki_lhn1.txt']
        with open('output_file_wiki.txt', 'w+') as output:
            reading = [open(filename) for filename in files]
            for lines in zip(*reading):
                print(' '.join([line.strip() for line in lines]), file = output)

    # to save the output file to a dataframe

    def similarity_df(self):
        global uni_df
        uni_df = pd.read_csv('output_file_wiki.txt', header = None)

        #uni_df.columns = ['node1', 'node2', 'cosine_score', 'jaccard_score', 'lhn1_score']


        # splitting uni_df into 5 columns

        uni_df[[0,1,2,3,4]] = pd.DataFrame([x.split() for x in uni_df[0].tolist()])

        # renaming the columns in uni_df
        uni_df.rename(columns={0: 'node1', 1: 'node2', 2: 'cosine_score', 3: 'jaccard_score', 4: 'lhn1_score'}, inplace=True)
        #le = preprocessing.LabelEncoder()
        #uni_df['cosine_score'] = le.fit_transform(uni_df['cosine_score'].astype(str))
        #uni_df = pd.to_numeric(uni_df['cosine_score'], errors='coerce')
        uni_df.to_csv(r'wiki_similarities.csv')

    def cosine_mat(self, dimension, output):
        global uni_df, remove_dupl
        remove_dupl = uni_df.drop_duplicates(keep = 'first')

        cos_sym = remove_dupl.pivot(index = 'node1', columns = 'node2', values = 'cosine_score')

        val = cos_sym.fillna(0)
        cos_mat = pd.DataFrame(val)
        cos_mat.to_csv(r'cosine_matrix_wiki.csv')
        cos_array = cos_mat.values

        #cos_array[cos_array > 0.25] = 0
        #global cosine_mat
        cosine_mat = coo_matrix(cos_array)
        cora_df = pd.read_csv('cosine_matrix_wiki.csv')
        #global cosine_mat

        mat_obj = cosine_mat.astype(float)

        u, _, _ = svds(mat_obj, dimension)
        print("Generating embeddings with cosine score")
        cos_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None, index=False, float_format='%8.7f', sep=' ')

        #cos_mat.data = np.where(cos_mat.data > 0.25, 0)






    def jaccard_mat(self, dimension, output):
        #global uni_df
        remove_dupl = uni_df.drop_duplicates(keep='first')
        jac_sym = remove_dupl.pivot(index='node1', columns='node2', values='jaccard_score')
        val = jac_sym.fillna(0)
        jac_mat = pd.DataFrame(val)
        jac_mat.to_csv(r'jaccard_matrix_wiki.csv')
        jac_array = jac_mat.values
        #global jaccard_mat
        jaccard_mat = coo_matrix(jac_array)
        cora_df = pd.read_csv('jaccard_matrix_wiki.csv')
        mat_obj = jaccard_mat.astype(float)

        u, _, _ = svds(mat_obj, dimension)
        print("Generating embeddings with jaccard score")
        jac_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None, index=False, float_format='%8.7f', sep=' ')

    def lhn1_mat(self, dimension, output):
        #global uni_df
        remove_dupl = uni_df.drop_duplicates(keep='first')
        #remove_dupl.to_csv(r'duplicates.csv')
        lhn1_sym = remove_dupl.pivot(index='node1', columns='node2', values='lhn1_score')
        val = lhn1_sym.fillna(0)
        lhn1_mat = pd.DataFrame(val)
        lhn1_mat.to_csv(r'lhn1_matrix_wiki.csv')

        lhn1_array = lhn1_mat.values
        #global lhn1_coomat

        lhn1_coomat = coo_matrix(lhn1_array)
        cora_df = pd.read_csv('lhn1_matrix_wiki.csv')
        mat_obj = lhn1_coomat.astype(float)

        u, _, _ = svds(mat_obj, dimension)
        print("Generating embeddings with lhn1 score")
        lhn1_embed = pd.DataFrame(np.hstack((cora_df.values[:, 0].reshape(-1, 1), u))).to_csv(output, header=None, index=False, float_format='%8.7f', sep=' ')

    def co_occurrence_matrix(self):
        # calculating co-occurrences on remove_dupl after removing all the duplicates
        uni_df['co-occurrence_count'] = uni_df.groupby(['node1', 'node2'])['node2'].transform('size')
        uni_df.to_csv(r'wiki_similarities.csv')
        remove_dupl = uni_df.drop_duplicates(keep='first')

        cora_coocc = remove_dupl.pivot(index='node1', columns='node2', values='co-occurrence_count')
        val = cora_coocc.fillna(0)

        coo_array = val.values
        global cooccur_mat
        cooccur_mat = coo_matrix(coo_array)

    def pmi_calculation(self,pmi_threshold = None):
        global cooccur_mat
        row_sum = np.squeeze(np.asarray(np.sum(cooccur_mat, axis=1)))
        total_sum = np.sum(row_sum)
        row_prob = row_sum / total_sum
        col_sum = np.squeeze(np.asarray(np.sum(cooccur_mat, axis=0)))
        col_prob = col_sum / total_sum
        global pmi_matrix
        pmi_matrix = np.log2(cooccur_mat.data / (total_sum * row_prob[cooccur_mat.row] * col_prob[cooccur_mat.col]))

        if pmi_threshold is None:
            pmi_rowcol = (cooccur_mat.row, cooccur_mat.col)
        else:
            thresholded_mat = (pmi_matrix >= pmi_threshold)
            pmi_matrix = pmi_matrix[thresholded_mat]
            pmi_rowcol = (cooccur_mat.row[thresholded_mat], cooccur_mat.col[thresholded_mat])
        return coo_matrix((pmi_matrix, pmi_rowcol), cooccur_mat.shape)

    #def run_svd_cosine(self, dimension, output):
        #global cosine_mat

        #mat_obj = cosine_mat.astype(float)


        #u, _, _ = svds(mat_obj, dimension)
        #print("Generating embeddings")
        #cos_embed = pd.DataFrame(u).to_csv(output, header=None, index=False, float_format='%8.7f', sep=' ')













        #cosine_mat = coo_matrix((val.values, (remove_dupl.node1, remove_dupl.node2)))
        #print(cosine_mat)

obj = Wiki()
obj.loading_file()
obj.random_walk(60)
obj.cosine()
obj.jaccard()
obj.lhn1()
obj.similarity_scores()
obj.similarity_df()
obj.cosine_mat(16,'cosine_embeddings_wiki.csv')
obj.jaccard_mat(16,'jaccard_embeddings_wiki.csv')
obj.lhn1_mat(16,'lhn1_embeddings_wiki.csv')
obj.co_occurrence_matrix()
obj.pmi_calculation(-2.5)


