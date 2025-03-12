from .base import BaseConnector
from rdflib_hdt import HDTStore
from rdflib import URIRef, Literal

import logging
logging.getLogger('rdflib').setLevel(logging.ERROR)

class HDTConnector(BaseConnector):
    def __init__(self, filename, skip_predicates=[]):
        self.filename = filename
        self.store = None
        self.skip_predicates = skip_predicates

    def get_node_edges(self, node):
        if self.store is None:
            self.store = HDTStore(self.filename)
        val = {}
        if node.startswith("http://") or node.startswith("_:"):
            try:
                res = list(self.store.hdt_document.search((URIRef(node), None, None))[0])
            except:
                res = []

            for r in res:
                try:
                    if r[1].toPython() not in self.skip_predicates:
                        if r[1].toPython() not in val:
                            val[r[1].toPython()] = []
                        if isinstance(r[2], Literal):
                            val[r[1].toPython()].append(r[2].n3().split('"')[1])
                        else:
                            val[r[1].toPython()].append(r[2].toPython())
                except:
                    continue

        return val


    def has_node_features(self):
        return False