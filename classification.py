# import statements

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import networkx as nx


# defining class with OneVsRestClassifier

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)

# defining the classification class

class Cora_classify():

    # defining the class constructor

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)





    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)

        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        results['acc'] = accuracy_score(Y, Y_)
        print('-------------------')
        print(results)
        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent,seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

# loading the graph again to fill the embedding matrix from node 0 to N

for name in ['wiki']:

    edge_path='%s-edgelist.txt'%name
    label_path='%s-label.txt'%name

#graph = "wiki-edgelist.txt"
#type_of_graph = nx.Graph()

# to read from the file

    new_graph = nx.read_edgelist(edge_path, create_using=nx.Graph(), nodetype=int)


# Loading the respective embeddings(cosine, jaccard, pmi, hpi, and hdi)

    #model = pd.read_csv("wiki-cosine-0.8-embed_9.csv", header=None)
    #model1 = pd.read_csv("wiki-jaccard-0.8-embed_9.csv", header=None)

    model3 = pd.read_csv("wiki-pmi-3-embed_9.csv", header=None)

    #model5 = pd.read_csv("wiki-hpi-0.8-embed_9.csv", header=None)
    #model6 = pd.read_csv("wiki-hdi-0.8-embed_9.csv", header=None)



# converting the embeddings dataframes to numpy nd arrays

    #model_arr = model.values
    #model_arr1 = model1.values

    model_arr3 = model3.values

    #model_arr5 = model5.values
    #model_arr6 = model6.values






#building the embedding matrix from the cosine-embedding matrix from node 0 to N.

    #embeddings = {}

    #no = new_graph.order()

    #for i in range(no):

        #embeddings[i] = model_arr[i]


#building the embedding matrix from the jaccard-embedding matrix from node 0 to N.



    #embeddings1 = {}

    #no1 = new_graph.order()

    #for i in range(no1):
        #embeddings1[i] = model_arr1[i]




#building the embedding matrix from the pmi-embedding matrix from node 0 to N.


    embeddings3 = {}

    no3 = new_graph.order()


    for i in range(no3):
        embeddings3[i] = model_arr3[i]



#building the embedding matrix from the hpi-embedding matrix from node 0 to N.


    #embeddings5 = {}

    #no5 = new_graph.order()


    #for i in range(no5):
        #embeddings5[i] = model_arr5[i]

#building the embedding matrix from the hdi-embedding matrix from node 0 to N.


    #embeddings6 = {}

    #no6 = new_graph.order()


    #for i in range(no6):
        #embeddings6[i] = model_arr6[i]





# reading the node label.


def read_node_label(filename=None,embeddings=None):
    fin = open(filename, 'r')
    X = []
    Y = []

    label = {}

    for line in fin:
        a = line.strip('\n').split(' ')
        label[a[0]] = a[1]


    fin.close()
    for i in embeddings:
        X.append(i)
        Y.append(label[str(i)])




    return X, Y



















