#import statements

import networkx as nx

#defining class
class Cora:
    def loading_file(self):
        graph = "cora-graph.txt"
        type_of_graph = nx.Graph()

        # to read from the file
        new_graph = nx.read_edgelist(graph, create_using=type_of_graph, nodetype=int)

        # Find the total number of nodes,edges and average degree of nodes
        print(nx.info(new_graph))

        #to check and remove self loops if any
        for u,v in new_graph.edges_iter():
            if u==v:
                new_graph.remove_edge(u,v)
        print("After checking for self-loops", new_graph.number_of_edges())

obj = Cora()
obj.loading_file()



