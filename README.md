# GraphFlex

**Flexible Framework for Graph Feature Engineering**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GraphFlex is a modular and extensible framework for graph-based feature engineering in Python. It allows seamless integration of graph datasets with traditional machine learning pipelines using familiar tools like `scikit-learn`.

> 🔗 Homepage & Documentation: [GraphFlex on GitHub](https://github.com/predict-idlab/graphflex)

## 📦 Installation

```bash
pip install graphflex
```
### Optional Dependencies

GraphFlex supports several optional extras. Install them with:

```bash
pip install "graphflex[dgl]"
pip install "graphflex[neo4j]"
pip install "graphflex[rdflib]"
pip install "graphflex[full]"  # all optional features
```

---

## 🔍 Example Usage

```python
# GraphFlex pipeline
from graphflex import GraphFlex
from graphflex.functions.postprocessing.filter import NonUniqueFeatureFilter
from graphflex.functions.feature import MeanStdFeature
from graphflex.functions.edgenode import NumericalEdgeNode

connect = Connector(...) #use defined connector here
gflex = GraphFlex(
    connector=connect,
    node_feature=MeanStdFeature(),
    edge_node_feature=NumericalEdgeNode(),
    post_processor=NonUniqueFeatureFilter()
)
nodes = ...
feature_matrix = gflex.fit_transform(nodes)
```
---

## ✨ Features

- Plug-and-play feature extraction for graph nodes
- Compatible with `scikit-learn` pipelines
- Support for multiple graph backends (DGL, RDFLib-HDT, Neo4j, ...)
- Built-in feature functions and postprocessing modules
- Easily extendable with custom logic
---

## 📚 Documentation

For full documentation, examples, and API reference, visit the [GraphFlex repository](https://github.com/predict-idlab/graphflex).

---

## ⚙ Dependencies

- Python ≥ 3.10
- `numpy`, `pandas`, `scikit-learn`, `tqdm`
- Optional: `dgl`, `torch`, `torchdata`, `rdflib-hdt`, `neo4j`, `PyYAML`, `pydantic`

---

## 👤 Author

**Bram Steenwinckel** – [bram.steenwinckel@ugent.be](mailto:bram.steenwinckel@ugent.be)

---

## 📄 License
This project is licensed under the MIT License.
