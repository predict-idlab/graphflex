import os
from tqdm import tqdm
from .base import BaseConnector

class EdgeListFileConnector(BaseConnector):
    def __init__(self, path):
        self.path = path
        self.feature_dct = {}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = "Node_"+str(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    if node_id not in self.feature_dct:
                        self.feature_dct[node_id] = {}

                    self.feature_dct[node_id]["type"] = node_type
                    for i in range(len(node_attr)):
                        self.feature_dct[node_id]["feat_"+str(i)] = node_attr[i]
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = "Node_"+str(node_id)
                    node_type = int(node_type)
                    if node_id not in self.feature_dct:
                        self.feature_dct[node_id] = {}

                    self.feature_dct[node_id]["type"] = node_type
                else:
                    raise Exception("Too few information to parse!")

        self.edge_dct = {}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = "Node_"+str(th[0]), "Node_"+str(th[1]), int(th[2]), float(th[3])
                if h_id not in self.edge_dct:
                    self.edge_dct[h_id] = {}
                if r_id not in self.edge_dct[h_id]:
                    self.edge_dct[h_id][r_id] = []
                self.edge_dct[h_id][r_id].append(t_id)

    def get_node_features(self, node):
        return self.feature_dct[node]

    def get_node_edges(self, node):
        if node in self.edge_dct:
            return self.edge_dct[node]
        else:
            return None

    def has_node_features(self):
        if self.feature_dct is not None:
            return True
        else:
            return False

    def load_labels(self, name):
        labels = {}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = "Node_"+str(th[0]), th[1], int(th[2]), list(
                    map(int, th[3].split(',')))
                nc = -1
                for label in node_label:
                    nc = max(nc, label)

                labels[node_id] = nc
        return labels

