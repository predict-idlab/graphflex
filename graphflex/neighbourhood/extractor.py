from .base import NeighbourhoodExtractor

class BFSNeighbourhoodExtractor(NeighbourhoodExtractor):
    def get_neighborhoods(self, noi, depth):
        neighbours = {}
        res = self.get_connector().get_node_edges(noi)
        if isinstance(res, list):
            neighbours[0] = [noi]
            for d in range(1, depth + 1):
                data = []
                for n in neighbours[d - 1]:
                    res2 = self.get_connector().get_node_edges(n)
                    if res2:
                        data.extend(res2)
                neighbours[d] = data

        if isinstance(res, dict):
            neighbours[0] = {'': [noi]}
            for d in range(1, depth + 1):
                next_level = {}
                for key, nodes in neighbours[d - 1].items():
                    for n in nodes:
                        res2 = self.get_connector().get_node_edges(n)
                        if res2:
                            for nkey, value in res2.items():
                                if key != '':
                                    composite_key = f"{key}¥{nkey}"
                                else:
                                    composite_key = f"{nkey}"
                                next_level[composite_key] = value

                neighbours[d] = next_level
        return neighbours

    def extract_features(self, neighbours, node_feature, edge_node_feature):
        all_feats = {}
        if len(neighbours) > 0:
            if isinstance(neighbours[0], list):
                for n in neighbours:
                    if self.get_connector().has_node_features():
                        node_features = [self.get_connector().get_node_features(i) for i in neighbours[n]]
                        all_feats.update(node_feature.extract(node_features, n, None))
                    all_feats.update(edge_node_feature.extract(neighbours[n], n, None))

            elif isinstance(neighbours[0], dict):
                for n in neighbours:
                    for k, v in neighbours[n].items():
                        if self.get_connector().has_node_features():
                            node_features = [self.get_connector().get_node_features(i) for i in v]
                            all_feats.update(node_feature.extract(node_features, n, k))
                        all_feats.update(edge_node_feature.extract(neighbours[n][k], n, k))

        return all_feats


