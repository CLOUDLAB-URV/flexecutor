# generate_graphs.py
import pickle
import networkx as nx
from lithops import Storage

BUCKET = "your-bucket-name"     # Replace with your bucket name
NODES = 100                     # Higher means more execution time
EDGE_PROB = 0.5
NUM_FUNCTIONS = 10

def gen_graphs(n):
    storage = Storage()
    storage.create_bucket(BUCKET)
    for i in range(n):
        G = nx.erdos_renyi_graph(NODES, EDGE_PROB)
        storage.put_object(BUCKET, f"graphs/graph{i}", pickle.dumps(G))
        del G

if __name__ == "__main__":
    gen_graphs(NUM_FUNCTIONS)
