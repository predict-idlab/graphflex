from .base import BaseConnector
import pandas as pd

class DGLConnector(BaseConnector):
    def __init__(self, dataset, feat_keyword='feat', heterogeneous=False):
        self.g = dataset[0]
        if feat_keyword is None:
            self.feature_dct = None
        else:
            self.feature_dct = pd.DataFrame(self.g.ndata[feat_keyword].numpy()).to_dict("index")
        if heterogeneous:
            self.edge_dct = {}
            for etype in self.g.canonical_etypes:
                rel_type = etype[1]
                src, dst = self.g.edges(etype=etype)
                src = src.numpy()
                dst = dst.numpy()
                # try:
                for i in range(len(src)):
                    if src[i] not in self.edge_dct:
                        self.edge_dct[src[i]] = {}
                    if rel_type not in self.edge_dct[src[i]]:
                        self.edge_dct[src[i]][rel_type] = []
                    self.edge_dct[src[i]][rel_type].append(dst[i])
        else:
            self.edge_dct = pd.DataFrame(self.g.edges()).T.astype(int).groupby(0)[1].apply(list).to_dict()
            iedge_dct = pd.DataFrame(self.g.edges()).T.astype(int).groupby(1)[0].apply(list).to_dict()
            self.edge_dct.update(iedge_dct)

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


class DGLMultiConnector(BaseConnector):
    def __init__(self, dataset, feat_keyword='feat', heterogeneous=False):
        self.edge_dct = {}
        self.feature_dct = {} if feat_keyword is not None else None

        graph_index = 0
        for g in dataset:
            if feat_keyword is not None:
                feature_data = pd.DataFrame(g.ndata[feat_keyword].numpy()).to_dict("index")
                for node_id, features in feature_data.items():
                    self.feature_dct[(graph_index, node_id)] = features

            if heterogeneous:
                for etype in g.canonical_etypes:
                    rel_type = etype[1]
                    src, dst = g.edges(etype=etype)
                    src = src.numpy()
                    dst = dst.numpy()

                    for i in range(len(src)):
                        src_key = (graph_index, src[i])
                        dst_node = (graph_index, dst[i])

                        if src_key not in self.edge_dct:
                            self.edge_dct[src_key] = {}
                        if rel_type not in self.edge_dct[src_key]:
                            self.edge_dct[src_key][rel_type] = []
                        self.edge_dct[src_key][rel_type].append(dst_node)
            else:
                edge_list = pd.DataFrame(g.edges()).T.astype(int)
                edge_dict = edge_list.groupby(0)[1].apply(list).to_dict()
                iedge_dict = edge_list.groupby(1)[0].apply(list).to_dict()

                for key, value in edge_dict.items():
                    key = (graph_index, key)
                    value = [(graph_index, v) for v in value]
                    if key in self.edge_dct:
                        self.edge_dct[key].extend(value)
                    else:
                        self.edge_dct[key] = value

                for key, value in iedge_dict.items():
                    key = (graph_index, key)
                    value = [(graph_index, v) for v in value]
                    if key in self.edge_dct:
                        self.edge_dct[key].extend(value)
                    else:
                        self.edge_dct[key] = value
            graph_index+=1

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
