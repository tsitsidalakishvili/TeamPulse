
import neo4j
from neo4j import GraphDatabase
import csv

class Neo4jManager:
    def __init__(self, uri, username, password):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self._driver.close()

    def create_node(self, label, properties):
        with self._driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} $properties)"
        tx.run(query, properties=properties)
        
    def create_node(self, label, properties):
        with self._driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

def create_nodes_from_csv(neo4j_manager, csv_file_path, label):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Assuming your CSV has columns like 'name', 'age', 'city'
            properties = {
                'name': row['name'],
                'age': int(row['age']),
                'city': row['city']
            }
            neo4j_manager.create_node(label, properties)

# Handle CSV upload and create nodes
if __name__ == "__main__":
    neo4j_manager = Neo4jManager("your_neo4j_uri", "your_neo4j_username", "your_neo4j_password")
    csv_file_path = "path_to_your_uploaded_csv_file.csv"
    label = "Person"  # Set the label for the nodes
    create_nodes_from_csv(neo4j_manager, csv_file_path, label)
    neo4j_manager.close()
