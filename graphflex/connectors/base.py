class BaseConnector(object):
    def get_node_features(self, node):
        raise NotImplementedError

    def get_node_edges(self, node):
        raise NotImplementedError

    def has_node_features(self):
        raise NotImplementedError
