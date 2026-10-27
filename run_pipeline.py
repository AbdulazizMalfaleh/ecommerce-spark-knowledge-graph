"""Run the full Bronze -> Silver -> Gold pipeline.

    python run_pipeline.py                 # all layers
    python run_pipeline.py --layers silver gold
"""
import argparse
import time

from src import bronze, gold, graph, silver
from src.spark_session import get_spark

LAYERS = {"bronze": bronze.run, "silver": silver.run, "gold": gold.run, "graph": graph.run}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    args = parser.parse_args()

    spark = get_spark()
    print(f"Spark {spark.version} | Spark UI: {spark.sparkContext.uiWebUrl}")

    total = time.perf_counter()
    for layer in args.layers:
        start = time.perf_counter()
        LAYERS[layer](spark)
        print(f"  ({layer} took {time.perf_counter() - start:.1f}s)")
    print(f"\nPipeline finished in {time.perf_counter() - total:.1f}s")

    spark.stop()


if __name__ == "__main__":
    main()
