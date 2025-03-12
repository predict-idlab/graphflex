from abc import ABC, abstractmethod
import numpy as np
import sys

class EdgeNode(ABC):
    @abstractmethod
    def extract(self, nodes_edge_lst, depth, rel):
        pass


def _isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

class NoneEdgeNode(EdgeNode):
    def extract(self, nodes_edge_lst, depth, rel):
        return {}

class NumericalEdgeNode(EdgeNode):
    def extract(self, nodes_edge_lst, depth, rel):
        all_feats = {}
        for el in nodes_edge_lst:
            if _isfloat(el):
                if rel is None:
                    all_feats[sys.intern(f'path/{depth}_real')] = np.float32(el)
                else:
                    all_feats[sys.intern(f'{rel}$(real_value)')] = np.float32(el)
            else:
                if rel is None:
                    all_feats[sys.intern(f'path/{depth}/{el}')] = np.bool_(1)
                else:
                    all_feats[sys.intern(f'{rel}${el}')] = np.bool_(1)
        return all_feats

