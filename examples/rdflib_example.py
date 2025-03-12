from graphflex import GraphFlex
from graphflex.connectors.sparql import RDFLibConnector
from graphflex.functions.edgenode import NumericalEdgeNode
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from sklearn.ensemble import ExtraTreesClassifier

connector = RDFLibConnector('../data/animals/animals.owl', 'xml')

pos = ["http://dl-learner.org/benchmark/dataset/animals#dog01",
       "http://dl-learner.org/benchmark/dataset/animals#dolphin01",
       "http://dl-learner.org/benchmark/dataset/animals#platypus01",
       "http://dl-learner.org/benchmark/dataset/animals#bat01"]

neg = ["http://dl-learner.org/benchmark/dataset/animals#trout01",
       "http://dl-learner.org/benchmark/dataset/animals#herring01",
       "http://dl-learner.org/benchmark/dataset/animals#shark01",
       "http://dl-learner.org/benchmark/dataset/animals#lizard01",
       "http://dl-learner.org/benchmark/dataset/animals#croco01",
       "http://dl-learner.org/benchmark/dataset/animals#trex01",
       "http://dl-learner.org/benchmark/dataset/animals#turtle01",
       "http://dl-learner.org/benchmark/dataset/animals#eagle01",
       "http://dl-learner.org/benchmark/dataset/animals#ostrich01",
       "http://dl-learner.org/benchmark/dataset/animals#penguin01"]

nodes = pos + neg
labels = [1 for _ in range(len(pos))] + [0 for _ in range(len(neg))]


gflex = GraphFlex(connector,
                  max_depth=10,
                  edge_node_feature=NumericalEdgeNode(),
                  post_processor=NonUniqueFeatureFilter(),
                  n_jobs=4,
                  verbose=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(nodes,labels,test_size=0.5, stratify=labels,random_state=42)

train_matrix = gflex.fit_transform(X_train)
print(train_matrix.shape)

clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(train_matrix, y_train)

print(clf.score(gflex.transform(X_test), y_test))