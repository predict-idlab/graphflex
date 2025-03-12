from .base import BaseConnector
from rdflib import Graph, Literal
import json

import logging
logging.getLogger('rdflib').setLevel(logging.ERROR)

class RDFLibConnector(BaseConnector):
    def __init__(self, filename, dataformat, skip_predicates=[]):
        self.g = Graph()
        self.g.parse(filename, format=dataformat)
        self.skip_predicates = skip_predicates

    def get_node_edges(self, node):
        val = {}
        if node.startswith("http://") or node.startswith("_:"):
            try:
                q_str = 'SELECT ?p ?o ?dt WHERE { BIND( IRI("' + node + '") AS ?s ) ?s ?p ?o. }'
                res = self.query(q_str)
            except:
                res = []

            for r in res:
                #try:
                    if r['p']['value'] not in self.skip_predicates:
                        if r['p']['value'] not in val:
                            val[r['p']['value']] = []
                        if isinstance(r['o']['value'], Literal):
                            val[r['p']['value']].append(r['o'].n3().split('"')[1])
                        else:
                            val[r['p']['value']].append(r['o']['value'])
                #except:
                 #   continue

        return val


    def has_node_features(self):
        return False

    def query(self, q_str):
        """
        Execute a query through RDFLib
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        res = self.g.query(q_str)
        return json.loads(res.serialize(format="json"))['results']['bindings']
