from abc import ABC, abstractmethod
from functools import partial
import multiprocessing as mp
from tqdm.auto import tqdm

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass
connector = None

class NeighbourhoodExtractor(ABC):
    @abstractmethod
    def get_neighborhoods(self, noi, depth):
        pass
    @abstractmethod
    def extract_features(self, neighbours, node_feature, edge_node_feature):
        pass

    def get_connector(self):
        return connector

    def init_worker(self, graph_connector):
        global connector
        connector = graph_connector

    def process_node(self, noi, depth, node_feature, edge_node_feature):
        neighbours = self.get_neighborhoods(noi, depth)
        features = self.extract_features(neighbours, node_feature, edge_node_feature)
        return noi, features

    def extract_neighborhoods(self, nodes, depth, connector, n_jobs, node_feature, edge_node_feature, verbose):
        all_data = {}
        partial_process_node = partial(self.process_node,
                                       depth=depth,
                                       node_feature=node_feature,
                                       edge_node_feature=edge_node_feature)
        if n_jobs > 1:
            with mp.Pool(processes=n_jobs, initializer=self.init_worker, initargs=(connector,)) as pool:
                results = list(tqdm(pool.imap(partial_process_node, nodes), total=len(nodes), desc="Extracting neighbourhoods", disable=not verbose))
                pool.close()
        else:
            results = []
            self.init_worker(connector)
            for n in tqdm(nodes, desc="Extracting neighbourhoods", disable=not verbose):
                results.append(
                    self.process_node(n,
                                      depth=depth,
                                      node_feature=node_feature,
                                      edge_node_feature=edge_node_feature)
                )

        for key, value in results:
            all_data[key] = value
        return all_data