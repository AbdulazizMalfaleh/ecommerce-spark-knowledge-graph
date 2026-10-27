"""Export the knowledge graph as CSV files in neo4j-admin bulk-import format.

    python -m src.export_neo4j

Writes data/graph/neo4j/*.csv plus import_command.txt with the exact command to
load everything into Neo4j (see README, "Explore in Neo4j").
"""
import shutil
import tempfile
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import GRAPH_DIR
from src.graph import load
from src.ontology import Ontology

OUT_DIR = GRAPH_DIR / "neo4j"
NEO4J_TYPES = {"string": "", "long": ":long", "int": ":int", "double": ":double",
               "boolean": ":boolean", "timestamp": ":datetime"}


def _write_single_csv(df: DataFrame, target: Path) -> None:
    """Spark writes a folder of part files; Neo4j wants one file per table."""
    with tempfile.TemporaryDirectory(dir=OUT_DIR) as tmp:
        (df.coalesce(1).write.mode("overwrite")
           .option("header", True)
           .option("escape", '"')                                  # Neo4j expects "" inside quoted values
           .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss'Z'")
           .csv(tmp))
        part = next(Path(tmp).glob("part-*.csv"))
        shutil.move(str(part), target)


def export(spark: SparkSession) -> None:
    onto = Ontology.load()
    nodes, edges = load(spark)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = []

    for name, ent in onto.entities.items():
        cols = [F.col("id").alias(f"id:ID({name})")]
        cols += [F.col(p).alias(f"{p}{NEO4J_TYPES[t]}") for p, t in ent.properties.items()]
        _write_single_csv(nodes[name].select(*cols), OUT_DIR / f"{name}.csv")
        args.append(f"--nodes={name}=/import/{name}.csv")
        print(f"  node  {name}.csv")

    for name, rel in onto.relations.items():
        cols = [F.col("src").alias(f":START_ID({rel.source})"), F.col("dst").alias(f":END_ID({rel.target})")]
        cols += [F.col(p).alias(f"{p}{NEO4J_TYPES[t]}") for p, t in rel.properties.items()]
        _write_single_csv(edges[name].select(*cols), OUT_DIR / f"{name}.csv")
        args.append(f"--relationships={name}=/import/{name}.csv")
        print(f"  edge  {name}.csv")

    # one line, so it pastes into cmd, PowerShell and bash alike
    command = "neo4j-admin database import full --overwrite-destination " + " ".join(args) + " neo4j\n"
    (OUT_DIR / "import_command.txt").write_text(command, encoding="utf-8")
    print(f"\nFiles and import command written to {OUT_DIR}")


if __name__ == "__main__":
    from src.spark_session import get_spark
    s = get_spark("export-neo4j")
    export(s)
    s.stop()
