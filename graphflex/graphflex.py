from graphflex.neighbourhood.extractor import BFSNeighbourhoodExtractor
from graphflex.functions.postprocessing.base import NoneProcessor
from graphflex.functions.edgenode import NoneEdgeNode
from graphflex.functions.feature import NoneFeature
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.auto import tqdm
import numpy as np

class GraphFlex(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        connector,
        max_depth=1,
        node_feature=NoneFeature(),
        edge_node_feature=NoneEdgeNode(),
        post_processor=NoneProcessor(),
        neighbourhood_extractor=BFSNeighbourhoodExtractor(),
        n_jobs=1,
        verbose=False
    ):
        self.connector = connector
        self.max_depth = max_depth
        self.col_idx = {}
        self.n_jobs = n_jobs
        self.node_feature = node_feature
        self.edge_node_feature = edge_node_feature
        self.post_processor = post_processor
        self.neighbourhood_extractor = neighbourhood_extractor
        self.verbose = verbose

    def _generate_features(self, nodes, col_idx=None):
        all_nbs = self.neighbourhood_extractor.extract_neighborhoods(nodes,
                                                           self.max_depth,
                                                           self.connector,
                                                           self.n_jobs,
                                                           self.node_feature,
                                                           self.edge_node_feature,
                                                           self.verbose)

        rows = list(all_nbs)
        cols = list({col for d in all_nbs.values() for col in d})
        row_idx = {row: i for i, row in enumerate(rows)}

        if col_idx is None:
            col_idx = {col: j for j, col in enumerate(cols)}

        row_indices = []
        col_indices = []
        values = []
        for row, col_dict in tqdm(all_nbs.items(), desc="Constructing matrix", disable=not self.verbose):
            i = row_idx[row]
            for col, value in col_dict.items():
                if col in col_idx:
                    row_indices.append(i)
                    col_indices.append(col_idx[col])
                    values.append(value)
        # Create the matrix and fill it in one vectorized step
        matrix = np.zeros((len(rows), len(col_idx)))
        matrix[np.array(row_indices), np.array(col_indices)] = values
        return matrix, {v: k for k, v in row_idx.items()}, col_idx

    def fit(self, nodes):
        matrix, row_idx, col_idx = self._generate_features(nodes)
        matrix, columns = self.post_processor.reduce(matrix, col_idx)
        self.col_idx = columns
        return self

    def transform(self, nodes):
        matrix, row_idx, _ = self._generate_features(nodes, self.col_idx)
        return matrix

    def fit_transform(self, nodes, y=None, **fit_params):
        matrix, row_idx, col_idx = self._generate_features(nodes)
        matrix, columns = self.post_processor.reduce(matrix, col_idx)
        self.col_idx = columns
        return matrix
