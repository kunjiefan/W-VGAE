import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize


def load_data():
    # load the data: x, tx, allx, graph
    names = ['allx_ct', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("../data/hprd-07/test1/NoElimilation/{}.pkl".format(names[i]))))
    allx, graph = tuple(objects)
    
    #deal with negative samples
    with open("../data/hprd-07/test1/NoElimilation/falseEdge.txt",'r') as f:
        lines = f.readlines()

    rows = []
    cols = []
    for line in lines:
        x,y = line.split()
        rows.append(int(x))
        cols.append(int(y))

    X = np.array(rows)
    Y = np.array(cols)
    falseEdges = np.vstack((X,Y)).transpose()

    
    features = normalize(allx)
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, falseEdges, graph
