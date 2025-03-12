from graphflex import GraphFlex
from graphflex.connectors.local import EdgeListFileConnector
from graphflex.functions.feature import MeanStdFeature
from graphflex.functions.edgenode import NumericalEdgeNode
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

connector = EdgeListFileConnector("../data/ACM/")

train_data = connector.load_labels("label.dat")
test_data = connector.load_labels("label.dat.test")

nodes_train = list(train_data.keys())
nodes_test = list(test_data.keys())

train_label = list([train_data[x] for x in nodes_train])
test_label = list([test_data[x] for x in nodes_test])

gflex = GraphFlex(connector, max_depth=1,
               node_feature=MeanStdFeature(),
               edge_node_feature=NumericalEdgeNode(),
               post_processor=NonUniqueFeatureFilter(),
               n_jobs=4,
               verbose=True)

train_matrix = gflex.fit_transform(nodes_train)

clf = LogisticRegression(max_iter=10 ** 20, C=10, tol=1e-7).fit(train_matrix,train_label)
y_pred = clf.predict(gflex.transform(nodes_test))

# Compute micro metrics
print('micro F1 score:', f1_score(test_label, y_pred, average='micro'))