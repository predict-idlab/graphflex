from graphflex import GraphFlex
from graphflex.connectors.dgl import DGLConnector
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from graphflex.functions.feature import MeanStdFeature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from dgl.data import CoraGraphDataset


dataset = CoraGraphDataset()

connector = DGLConnector(dataset)
nodes = dataset[0].nodes().numpy()
labels = dataset[0].ndata["label"].numpy()

pipe = Pipeline([('graphflex', GraphFlex(connector,
                                         1,
                                         node_feature=MeanStdFeature(),
                                         post_processor=NonUniqueFeatureFilter(),
                                         n_jobs=8,
                                         verbose=False)),
                 ('lr', LogisticRegression(C=10))])

cv = StratifiedKFold(n_splits=3)
scores = cross_val_score(pipe, nodes, labels, cv = cv)
print(scores)