import cPickle as pkl
from preprocessing import construct_optimizer_list, make_test_edges
from input_data import load_data
import scipy.sparse as sp
import numpy as np

from trainGcn import train_gcn
from trainNN import generate_data,train_nn

def main():
    adj, features, falseEdges, graph = load_data()

    weight_rate = 5
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = make_test_edges(weight_rate, adj, falseEdges)

    emb = train_gcn(graph,features,adj_train,train_edges,train_edges_false,test_edges,test_edges_false)

    X_train,Y_train = generate_data(emb, train_edges, train_edges_false)
    X_test,Y_test = generate_data(emb, test_edges, test_edges_false)

    acc = train_nn(X_train,Y_train,X_test,Y_test)

    
main()
