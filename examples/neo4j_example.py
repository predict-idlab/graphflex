from graphflex import GraphFlex
from graphflex.functions.feature import MeanStdFeature
from graphflex.connectors.cypher import Neo4jConnector
from graphflex.functions.edgenode import NumericalEdgeNode
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# We preloaded the Cora dataset in a neo4j database with some test credentials
# To load the cora dataset yourself, you can follow the following documented steps here:
# https://neo4j.com/docs/graph-data-science-client/current/tutorials/ml-pipelines-node-classification/

connector = Neo4jConnector(uri="bolt://localhost:7687",
                           username="test",
                           password="test1234",
                           database="test",
                           id_type="extId",
                           skip_predicates=["subject"]
            )

query = f"""
            MATCH (n:Paper)
            RETURN n.extId AS id, n.subject AS label
        """
q_result = connector.execute_query(query)
nodes = [record['id'] for record in q_result]

gflex = GraphFlex(connector,
               max_depth=2,
               node_feature=MeanStdFeature(),
               edge_node_feature=NumericalEdgeNode(),
               post_processor=NonUniqueFeatureFilter(),
               n_jobs=4,
               verbose=True
        )

X = gflex.fit_transform(nodes)
y = np.array([record['label'] for record in q_result])
print(X.shape)

accuracies = []
fold_nr = 0

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = ExtraTreesClassifier()
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"-- Accuracies fold {fold_nr}: {accuracy}")
    accuracies.append(accuracy)
    fold_nr += 1

print(f"Mean accuracy: {np.mean(accuracies)}")
