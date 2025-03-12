from graphflex import GraphFlex
from graphflex.connectors.hdt import HDTConnector
from graphflex.functions.edgenode import NumericalEdgeNode
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train_data = pd.read_csv("../data/BGS/train.tsv", sep="\t")
train_entities = [entity for entity in train_data["rock"]]
train_label = list(train_data["label_lithogenesis"])

test_data = pd.read_csv("../data/BGS/test.tsv", sep="\t")
test_entities = [entity for entity in test_data["rock"]]
test_label = list(test_data["label_lithogenesis"])

connector = HDTConnector("../data/BGS/BGS.hdt",
                         skip_predicates=['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'])
gflex = GraphFlex(connector,
                  max_depth=2,
                  edge_node_feature=NumericalEdgeNode(),
                  post_processor=NonUniqueFeatureFilter(),
                  n_jobs=4,
                  verbose=True)

train_matrix = gflex.fit_transform(train_entities)

clf = RandomForestClassifier(n_estimators=1000).fit(train_matrix, train_label)
y_pred = clf.predict(gflex.transform(test_entities))

# Compute micro metrics
print('Accuracy score:', accuracy_score(test_label, y_pred))
