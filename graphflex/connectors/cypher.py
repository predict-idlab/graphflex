from .base import BaseConnector
from neo4j import GraphDatabase

class Neo4jConnector(BaseConnector):
    def __init__(self, skip_predicates=[], uri="bolt://localhost:7687", username="neo4j", password="neo4j", database="neo4j", id_type="extId"):
        # Create the driver instance
        self.driver = None
        self.skip_predicates = skip_predicates
        self.id_type = id_type
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

    def get_node_edges(self, node):
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), database=self.database)
        query = f"""
                        MATCH (n {{{self.id_type}: {node}}})-[r]->(o)
                        RETURN DISTINCT type(r) AS relationship, o.{self.id_type} AS target
                    """

        with self.driver.session() as session:
            result = session.run(query)
            val = {}
            for record in result:
                relationship = record['relationship']
                target = record['target']

                if relationship not in []:
                    if relationship not in val:
                        val[relationship] = []
                    val[relationship].append(target)
        return val

    def get_node_features(self, node):
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), database=self.database)

        query = f"""
                        MATCH (n {{{self.id_type}: {node}}})
                        RETURN properties(n) AS props
                    """
        with self.driver.session() as session:
            result = session.run(query)
            props_list = []
            for record in result:
                props_dct = record['props']  # Extract the properties dictionary
                props_list = {}
                for x in props_dct:
                    if x != self.id_type and x not in self.skip_predicates:
                        if isinstance(props_dct[x], list):
                            props_list.update({x+'_'+str(i): props_dct[x][i] for i in range(len(props_dct[x]))})
                        else:
                            props_list[x] = props_dct[x]
        return props_list


    def execute_query(self, query):
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), database=self.database)
        with driver.session() as session:
            return list(session.run(query))

    def has_node_features(self):
        return True