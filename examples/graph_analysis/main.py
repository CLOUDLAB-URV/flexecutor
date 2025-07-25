from lithops import FunctionExecutor
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.workflow.stage import Stage
from flexecutor.storage.storage import FlexData

from functions import compute_pagerank, community_detection, dijkstra_analysis

if __name__ == "__main__":
    # TODO: Since this pipeline will be used to benchmark and test performance, we assume that any pagerank will work for the dijkstra analysis.
    # Future versions of this pipeline may include naming conventions or metadata to ensure that pagerank files correspond to their graphs. 
    @flexorchestrator(bucket="your-bucket-name") # Replace with your bucket name           
    def main():
        dag = DAG("graph-analysis")

        data_graphs = FlexData("graphs")
        data_pagerank = FlexData("pagerank")
        data_communities = FlexData("communities")
        data_dijkstra = FlexData("dijkstra")

        stage_pagerank = Stage(
            stage_id="pagerank",
            func=compute_pagerank,
            inputs=[data_graphs],
            outputs=[data_pagerank],
        )

        stage_community = Stage(
            stage_id="community",
            func=community_detection,
            inputs=[data_graphs],
            outputs=[data_communities]
        )

        stage_dijkstra = Stage(
            stage_id="dijkstra",
            func=dijkstra_analysis,
            inputs=[data_graphs, data_pagerank],
            outputs=[data_dijkstra]
        )

        stage_pagerank >> stage_dijkstra

        dag.add_stages([stage_pagerank, stage_community, stage_dijkstra])

        executor = DAGExecutor(
            dag,
            executor=FunctionExecutor(log_level="INFO"),
        )

        executor.execute()
        executor.shutdown()

    main()
