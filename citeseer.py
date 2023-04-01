import os
import os.path as osp
import pickle
import numpy as np
import itertools
import scipy.sparse as sp
import urllib  # import urllib.request ???
from collections import namedtuple
import networkx as nx
Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])


class CiteseerData(object):
    # download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    filenames = ["ind.citeseer.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="data", rebuild=False):


        self.data_root = data_root

        save_file = osp.join(self.data_root, "processed_citeseer.pkl")




        # self.maybe_download()
        self._data = self.process_data()

        with open(save_file, "wb") as f:
            pickle.dump(self.data, f)
        print("Cached file: {}".format(save_file))

    @property
    def data(self):

        return self._data

    def process_data(self):

        print("Process data ...")

        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, name)) for name in
                                                       self.filenames]

        # max_index = ally.shape[0] + ty.shape[0]
        # test_index = test_index[test_index < max_index]

        s = test_index.min()
        t = test_index.max()
        tx_zero = np.zeros(tx.shape[1], dtype=np.float).reshape(1, -1)
        ty_zero = np.zeros(ty.shape[1]).reshape(1, -1)
        for i in range(s, t + 1):
            if i not in test_index:
                arr_i = np.array(i).reshape(1, )
                test_index = np.concatenate((test_index, arr_i), axis=0)
                tx = np.concatenate((tx, tx_zero), axis=0)
                ty = np.concatenate((ty, ty_zero), axis=0)


        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        features = sp.vstack((allx, tx)).tolil()
        features[test_index, :] = features[sorted_test_index, :]
        x = np.concatenate((allx, tx), axis=0)

        y = np.concatenate((ally, ty), axis=0)


        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]

        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)

        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # for key in list(adjacency_dict):
        #     if key >= max_index:
        #         adjacency_dict.pop(key)
        #     else:
        #         adjacency_dict[key] = [v for k, v in enumerate(adjacency_dict[key]) if v < max_index]


        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adj.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())
        y_train = np.zeros(y.shape)
        y_val = np.zeros(y.shape)
        y_test = np.zeros(y.shape)
        y_train[train_mask, :] = y[train_mask, :]
        y_val[val_mask, :] = y[val_mask, :]
        y_test[test_mask, :] = y[test_mask, :]
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        # return Data(x=x, y=y, adjacency_dict=adjacency_dict,
        #             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def read_data(path):
        name = osp.basename(path)


        if name == "ind.citeseer.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"))
            out = out.toarray() if hasattr(out, "toarray") else out
            return out
