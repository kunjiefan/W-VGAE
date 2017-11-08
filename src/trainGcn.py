from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import cPickle as pkl

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, construct_optimizer_list


def train_gcn(graph, features,adj_train, train_edges, train_edges_false, test_edges, test_edges_false):
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 96, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 48, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
    flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
    flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

    model_str = FLAGS.model
    dataset_str = FLAGS.dataset

    #1-dim index array
    mask_index = construct_optimizer_list(graph, train_edges, train_edges_false)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float64),
        'adj': tf.sparse_placeholder(tf.float64),
        'adj_orig': tf.sparse_placeholder(tf.float64),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)


    #pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    #norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    pos_weight = 1
    norm = 1

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          mask=mask_index)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           mask=mask_index)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def get_roc_score(edges_pos, edges_neg, emb=None):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def relu(x):
            return max(x,0.0)
    
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)
        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)

        preds_all, labels_all  = [],[]
        for x in edges_pos:
            preds_all.append(adj_rec[x[0],x[1]])
        for x in edges_neg:
            preds_all.append(adj_rec[x[0],x[1]])
    
        preds_all = np.array(preds_all)
        labels_all = np.hstack((np.ones(edges_pos.shape[0]),np.full(edges_neg.shape[0],-1)))
        

        #roc_score = roc_auc_score(labels_all, preds_all)
        #ap_score = average_precision_score(labels_all, preds_all)
        #print("roc=","{:.5f}".format(roc_score),"ap=","{:.5f}".format(ap_score))



    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        #print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(avg_cost), "train_accuracy=", "{:.5f}".format(avg_accuracy))
        #get_roc_score(test_edges, test_edges_false)


    print("Optimization Finished!")
    #get_roc_score(test_edges, test_edges_false)

    emb = sess.run(model.z_mean,feed_dict=feed_dict)
    return emb

