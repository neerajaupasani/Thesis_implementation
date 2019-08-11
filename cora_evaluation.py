#import statements

import cora_classification
from cora_classification import embeddings,embeddings1,embeddings2, embeddings3
import cora_lp
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings





class Eval:


    # node classification for evaluating the embeddings generated from the cosine matrix.

    def cosine_classification(self,embeddings, label_path, name):

        X, Y = cora_classification.read_cora_node_label(label_path,embeddings)

        f_c=open('cora_nc_cos.txt', 'w')

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = cora_classification.Cora_classify(embeddings=embeddings,clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' '+ str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the jaccard matrix.

    def jaccard_classification(self,embeddings, label_path, name):

        X, Y = cora_classification.read_cora_node_label(label_path,embeddings)

        f_c=open('cora_nc_jac.txt', 'w+')

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = cora_classification.Cora_classify(embeddings=embeddings,clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' '+ str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the lhn1 matrix.

    def lhn1_classification(self,embeddings, label_path, name):

        X, Y = cora_classification.read_cora_node_label(label_path,embeddings)

        f_c=open('cora_nc_lhn1.txt', 'w+')

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = cora_classification.Cora_classify(embeddings=embeddings,clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' '+ str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # node classification for evaluating the embeddings generated from the pmi matrix.

    def pmi_classification(self,embeddings, label_path, name):

        X, Y = cora_classification.read_cora_node_label(label_path,embeddings)

        f_c=open('cora_nc_pmi.txt', 'w+')

        for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

               print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
               clf = cora_classification.Cora_classify(embeddings=embeddings,clf=LogisticRegression())
               results= clf.split_train_evaluate(X, Y, tr_frac)
               warnings.filterwarnings('ignore')

               for avg in [ "macro"]:
                 f_c.write(name + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
                 f_c.write('\n')

    # link prediction for evaluating the embeddings generated from the cosine matrix.

    def link_prediction_cosine(self, edge_file, embeddings, size, name):

            clf = cora_lp.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('cora_lp_cosine_%d.txt' % size, 'w')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the jaccard matrix.

    def link_prediction_jaccard(self, edge_file, embeddings, size, name):

            clf = cora_lp.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('cora_lp_jaccard_%d.txt' % size, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the lhn1 matrix.

    def link_prediction_lhn1(self, edge_file, embeddings, size, name):

            clf = cora_lp.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('cora_lp_lhn1_%d.txt' % size, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')

    # link prediction for evaluating the embeddings generated from the pmi matrix.

    def link_prediction_pmi(self, edge_file, embeddings, size, name):

            clf = cora_lp.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
            auc=clf.predict(size)

            functions = ["hadamard","average","l1","l2"]

            f_l = open('cora_lp_pmi_%d.txt' % size, 'w+')

            for i in functions:
                print(i, '%.3f' % np.mean(auc[i]))

                f_l.write(name + ' ' + str(size) + ' ' + str(i) + ' ' + str('%.3f' % np.mean(auc[i])))
                f_l.write('\n')





eval = Eval()
eval.cosine_classification(embeddings=embeddings,label_path='cora-label.txt',name='Evaluating the embeddings generated from the cosine matrix')
eval.jaccard_classification(embeddings=embeddings1,label_path='cora-label.txt',name='Evaluating the embeddings generated from the jaccard matrix')
eval.lhn1_classification(embeddings=embeddings2,label_path='cora-label.txt',name='Evaluating the embeddings generated from the lhn1 matrix')
eval.pmi_classification(embeddings=embeddings3,label_path='cora-label.txt',name='Evaluating the embeddings generated from the pmi matrix')

eval.link_prediction_cosine('cora-edgelist.txt',embeddings=embeddings,size=128,name='Evaluating the embeddings generated from the cosine matrix')
eval.link_prediction_jaccard('cora-edgelist.txt',embeddings=embeddings1,size=128,name='Evaluating the embeddings generated from the jaccard matrix')
eval.link_prediction_lhn1('cora-edgelist.txt',embeddings=embeddings2,size=128,name='Evaluating the embeddings generated from the lhn1 matrix')
eval.link_prediction_pmi('cora-edgelist.txt',embeddings=embeddings3,size=128,name='Evaluating the embeddings generated from the pmi matrix')








