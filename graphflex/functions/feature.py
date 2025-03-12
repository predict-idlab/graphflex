from abc import ABC, abstractmethod
import numpy as np
import sys

class NodeFeature(ABC):
    @abstractmethod
    def extract(self, nodes_feature_lst, depth, rel):
        pass

class NoneFeature(NodeFeature):
    def extract(self, nodes_feature_lst, depth, rel):
        return {}

class MeanStdFeature(NodeFeature):
    def extract(self, nodes_feature_lst, depth, rel):
        all_feats = {}
        try:
            values = np.array([[d[key] for d in nodes_feature_lst] for key in nodes_feature_lst[0].keys()])
            means = np.mean(values, axis=1)
            stds = np.std(values, axis=1)

            for key in range(len(nodes_feature_lst[0].keys())):
                if rel is None:
                    all_feats[sys.intern(f'mean/{depth}/{key}')] = np.float16(means[key])
                    all_feats[sys.intern(f'std/{depth}/{key}')] = np.float16(stds[key])
                elif rel == '':
                    all_feats[sys.intern(f'mean/{key}')] = np.float16(means[key])
                    all_feats[sys.intern(f'std/{key}')] = np.float16(stds[key])
                else:
                    all_feats[sys.intern(f'mean/{rel}/{key}')] = np.float16(means[key])
                    all_feats[sys.intern(f'std/{rel}/{key}')] = np.float16(stds[key])
            return all_feats
        except Exception as e:
            return {}