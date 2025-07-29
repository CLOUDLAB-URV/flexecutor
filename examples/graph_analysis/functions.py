import pickle
import networkx as nx
import community.community_louvain as community_louvain
from lithops.storage import Storage
from flexecutor import StageContext

N_DIJKSTRA = 150

def compute_pagerank(ctx: StageContext):

    graph_path = ctx.get_input_paths("graphs")

    for _,graph_path in enumerate(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        pagerank = nx.pagerank(graph, alpha=0.99)

        pr_path = ctx.next_output_path("pagerank")

        with open(pr_path, "wb") as f:
            pickle.dump(pagerank, f)

def community_detection(ctx: StageContext):

    graph_path = ctx.get_input_paths("graphs")

    for _,graph_path in enumerate(graph_path):

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        communities = community_louvain.best_partition(graph)
        com_path = ctx.next_output_path("communities")

        with open(com_path, "wb") as f:
            pickle.dump(communities, f)

def dijkstra_analysis(ctx: StageContext):
    graph_paths = ctx.get_input_paths("graphs")
    pagerank_paths = ctx.get_input_paths("pagerank")

    if len(graph_paths) != len(pagerank_paths):
        raise RuntimeError("Mismatch between number of graphs and pagerank files.")

    for graph_path, pagerank_path in zip(graph_paths, pagerank_paths):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        with open(pagerank_path, "rb") as f:
            pagerank = pickle.load(f)

        important_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:N_DIJKSTRA]

        shortest_paths = {
            node: nx.single_source_dijkstra_path(graph, node)
            for node in important_nodes
        }

        out_path = ctx.next_output_path("dijkstra")

        with open(out_path, "wb") as f:
            pickle.dump(shortest_paths, f)