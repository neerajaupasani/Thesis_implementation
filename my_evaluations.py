#import statements


from classification import read_node_label, Cora_classify
import prediction
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings


def node_classification(embeddings, label_path, name, emb, similarity, threshold):

    X, Y = read_node_label(label_path,embeddings)

    f_c=open('%s-%s-%s-classification.txt'%(emb,name,threshold), 'w+')

    for tr_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

           print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
           clf = Cora_classify(embeddings=embeddings, clf=LogisticRegression())
           results= clf.split_train_evaluate(X, Y, tr_frac)

           for avg in [ "macro"]:
             f_c.write(emb+' '+ name + ' ' + similarity + ' ' + str(threshold) + ' ' + str(tr_frac)+ ' '+avg+ ' '+ str('%0.5f'%results[avg]))
             f_c.write('\n')


def link_pred( edge_file, embeddings, name, size, emb, similarity, threshold):

        clf = prediction.Cora_linkprediction(embeddings=embeddings, edge_file= edge_file)
        auc=clf.predict(size)

        functions = ["hadamard","average","l1","l2"]

        f_l = open('%s-%s-%s-linkpred.txt' % (emb, name, threshold), 'w+')

        for i in functions:
            print(i, '%.3f' % np.mean(auc[i]))

            f_l.write(emb+' '+ name + ' ' + similarity + ' ' + str(threshold) + ' ' + str(size) + ' ' + str(i) + ' AUC: ' + str('%.3f' % np.mean(auc[i])))
            f_l.write('\n')




def plot_embeddings( embeddings,label_path, name):
    X, Y = read_node_label(label_path,embeddings)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1],label=c)  # c=node_colors)

    plt.axis('off')
    plt.legend(loc= 'upper right', prop={'size': 15}, bbox_to_anchor=(1.15, 1), ncol=1)
    #plt.title('%s graph '%name)
    plt.savefig('%s_vis.pdf'%(name),  bbox_inches='tight',dpi=100)




