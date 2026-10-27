"""Builds a SparkSession tuned for running locally on a laptop."""
import os
import sys

from pyspark.sql import SparkSession

from src import win_env

# Windows: point Spark at Java 17/21 and the Hadoop helper files (no-op elsewhere).
win_env.configure()

# Make Spark's Python workers use the same interpreter as this script (the venv).
# Without this, Windows often fails with "Python worker failed to connect back"
# because it looks for a "python3" executable that doesn't exist there.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark(app_name: str = "Ecommerce-Pipeline") -> SparkSession:
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                                   # use every CPU core
        .config("spark.driver.memory", driver_memory)
        # Default is 200 shuffle partitions -> far too many tiny tasks on one machine.
        .config("spark.sql.shuffle.partitions", "8")
        # Adaptive Query Execution: re-optimises the plan at runtime using real stats.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
