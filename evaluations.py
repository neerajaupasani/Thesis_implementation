#import statements

import classification
from classification import embeddings,embeddings1,embeddings2, embeddings3, embeddings4, embeddings5, embeddings6
import prediction
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings





class Eval:


    # node classification for evaluating the embeddings generated from the cosine matrix.

    def cosine_classification(self,embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c=open('cora_nc_cos_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the jaccard matrix.

    def jaccard_classification(self,embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c=open('cora_nc_jac_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the lhn1 matrix.

    def lhn1_classification(self,embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c=open('cora_nc_lhn1_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the sorensen matrix.

    def sorensen_classification(self, embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c = open('cora_nc_sorensen_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

            print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
            clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
            results = clf.split_train_evaluate(X, Y, tr_frac)
            warnings.filterwarnings('ignore')

            for avg in ["macro"]:
                f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac) + ' ' + avg + ' ' + str('%0.5f' % results[avg]))
                f_c.write('\n')

    # node classification for evaluating the embeddings generated from the hpi matrix.

    def hpi_classification(self, embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c = open('cora_nc_hpi_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

            print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
            clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
            results = clf.split_train_evaluate(X, Y, tr_frac)
            warnings.filterwarnings('ignore')

            for avg in ["macro"]:
                f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac) + ' ' + avg + ' ' + str('%0.5f' % results[avg]))
                f_c.write('\n')

    # node classification for evaluating the embeddings generated from the hdi matrix.

    def hdi_classification(self, embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c = open('cora_nc_hdi_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

            print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
            clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
            results = clf.split_train_evaluate(X, Y, tr_frac)
            warnings.filterwarnings('ignore')

            for avg in ["macro"]:
                f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac) + ' ' + avg + ' ' + str('%0.5f' % results[avg]))
                f_c.write('\n')

    # node classification for evaluating the embeddings generated from the pmi matrix.

    def pmi_classification(self,embeddings, label_path, name, similarity, threshold):

        X, Y = classification.read_cora_node_label(label_path, embeddings)

        f_c=open('cora_nc_pmi_%s.txt' % threshold, 'w+')

        for tr_frac in [0.8]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = classification.Cora_classify(embeddings=embeddings, clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # link prediction for evaluating the embeddings generated from the cosine matrix.

    def link_prediction_cosine(self, edge_file, embeddings, size, name, similarity, threshold):

            clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('wiki_lp_cosine_%s.txt' % threshold, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the jaccard matrix.

    def link_prediction_jaccard(self, edge_file, embeddings, size, name, similarity, threshold):

            clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('wiki_lp_jaccard_%s.txt' % threshold, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the lhn1 matrix.

    def link_prediction_lhn1(self, edge_file, embeddings, size, name, similarity, threshold):

            clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('wiki_lp_lhn1_%s.txt' % threshold, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the sorensen matrix.

    def link_prediction_sorensen(self, edge_file, embeddings, size, name, similarity, threshold):

        clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file=edge_file)
        auc = clf.predict(size)

        functions = ["hadamard", "average", "l1", "l2"]

        f_l = open('wiki_lp_sorensen_%s.txt' % threshold, 'w+')

        for i in functions:
            print(i, '%.3f' % np.mean(auc[i]))

            f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
            f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the hpi matrix.

    def link_prediction_hpi(self, edge_file, embeddings, size, name, similarity, threshold):

        clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file=edge_file)
        auc = clf.predict(size)

        functions = ["hadamard", "average", "l1", "l2"]

        f_l = open('wiki_lp_hpi_%s.txt' % threshold, 'w+')

        for i in functions:
            print(i, '%.3f' % np.mean(auc[i]))

            f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
            f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the hdi matrix.

    def link_prediction_hdi(self, edge_file, embeddings, size, name, similarity, threshold):

        clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file=edge_file)
        auc = clf.predict(size)

        functions = ["hadamard", "average", "l1", "l2"]

        f_l = open('wiki_lp_hdi_%s.txt' % threshold, 'w+')

        for i in functions:
            print(i, '%.3f' % np.mean(auc[i]))

            f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
            f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the pmi matrix.

    def link_prediction_pmi(self, edge_file, embeddings, size, name, similarity, threshold):

            clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('cora_lp_pmi_%s.txt' % threshold, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + similarity + ' ' + threshold + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')





eval = Eval()
eval.cosine_classification(embeddings=embeddings,label_path='cora-label.txt',name='Cora', similarity='cosine', threshold='alpha 0.0')
eval.jaccard_classification(embeddings=embeddings1,label_path='cora-label.txt',name='Cora', similarity='jaccard', threshold='alpha 0.0')
eval.lhn1_classification(embeddings=embeddings2,label_path='cora-label.txt',name='Cora', similarity='lhn1', threshold='alpha 0.0')
eval.sorensen_classification(embeddings=embeddings4,label_path='cora-label.txt',name='Cora', similarity='sorensen', threshold='alpha 0.0')
eval.hpi_classification(embeddings=embeddings5,label_path='cora-label.txt',name='Cora', similarity='hpi', threshold='alpha 0.0')
eval.hdi_classification(embeddings=embeddings6,label_path='cora-label.txt',name='Cora', similarity='hdi', threshold='alpha 0.0')
eval.pmi_classification(embeddings=embeddings3,label_path='citeseer-label.txt',name='Citeseer', similarity='pmi', threshold='alpha -3.0')

eval.link_prediction_cosine('cora-edgelist.txt',embeddings=embeddings,size=128,name='Cora', similarity='cosine', threshold='alpha 0.0')
eval.link_prediction_jaccard('cora-edgelist.txt',embeddings=embeddings1,size=128,name='Cora', similarity='jaccard', threshold='alpha 0.0')
eval.link_prediction_lhn1('cora-edgelist.txt',embeddings=embeddings2,size=128,name='Cora', similarity='lhn1', threshold='alpha 0.0')
eval.link_prediction_sorensen('cora-edgelist.txt',embeddings=embeddings4,size=128,name='Cora', similarity='sorensen', threshold='alpha 0.0')
eval.link_prediction_hpi('cora-edgelist.txt',embeddings=embeddings5,size=128,name='Cora', similarity='hpi', threshold='alpha 0.0')
eval.link_prediction_hdi('cora-edgelist.txt',embeddings=embeddings6,size=128,name='Cora', similarity='hdi', threshold='alpha 0.0')
eval.link_prediction_pmi('citeseer-edgelist.txt',embeddings=embeddings3,size=128,name='Citeseer', similarity='pmi', threshold='alpha -3.0')








