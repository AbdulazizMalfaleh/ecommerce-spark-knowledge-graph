"""Check that this machine can run the pipeline.   Run:  python check_setup.py

On Windows it also sets things up: it finds Java 17/21 even if an old Java is
first on PATH, and downloads the two Hadoop helper files into tools\\hadoop.
Then it starts Spark and writes/reads a small Parquet file end to end.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src import win_env  # noqa: E402

OK, FAIL = "[ OK ]", "[FAIL]"
problems = 0


def report(ok: bool, msg: str, fix: str = "") -> bool:
    global problems
    print(f"{OK if ok else FAIL} {msg}")
    if not ok:
        problems += 1
        if fix:
            print(f"       -> {fix}")
    return ok


# 1. Python
v = sys.version_info
report((3, 10) <= (v.major, v.minor) <= (3, 13), f"Python {v.major}.{v.minor}",
       "Use Python 3.10 - 3.13 (PySpark 4 does not support newer versions yet).")
report(sys.prefix != sys.base_prefix, "running inside a virtual environment",
       "Recommended: python -m venv .venv, then activate it (.venv\\Scripts\\activate on Windows).")

# 2. Windows: find Java 17/21 and get the Hadoop helper files
if win_env.IS_WINDOWS:
    try:
        used = win_env.configure(download=True)
    except RuntimeError as e:
        used = {}
        report(False, "Hadoop helper files", f"{e}. Check your internet connection and run this again.")
    report("JAVA_HOME" in used, f"Java 17/21 found: {used.get('JAVA_HOME', 'none')}",
           "Install Temurin 21 (.msi, Windows x64) from adoptium.net, then run this again.")
    if "HADOOP_HOME" in used:
        report(True, f"Hadoop helper files: {used['HADOOP_HOME']}\\bin")

# 3. Java actually used by Spark
java_home = os.environ.get("JAVA_HOME")
java = (str(Path(java_home) / "bin" / ("java.exe" if win_env.IS_WINDOWS else "java"))
        if java_home else shutil.which("java"))
if report(bool(java) and Path(java).exists(), f"Java executable: {java}",
          "Install Java 17 or 21 (e.g. Temurin 21 from adoptium.net)."):
    out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    major = int(m.group(1)) if m else 0
    if major == 1 and m.group(2):          # Java 8 and older report "1.8.0_xxx"
        major = int(m.group(2))
    report(major in (17, 21), f"Java version {major}", "Spark 4 needs Java 17 or 21.")

# 4. PySpark
try:
    import pyspark
    report(True, f"PySpark {pyspark.__version__}")
except ImportError:
    report(False, "PySpark installed", "pip install -r requirements.txt")
    sys.exit(1)

# 5. End-to-end: start Spark, write and read Parquet
if problems:
    print(f"\n{problems} problem(s) above. Fix them and run this again.")
    sys.exit(1)

print("\nStarting Spark (first start can take ~20s) ...")
from src.spark_session import get_spark  # noqa: E402

try:
    spark = get_spark("setup-check")
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "t.parquet")
        spark.range(1000).selectExpr("id", "id % 7 AS k").write.parquet(path)
        n = spark.read.parquet(path).groupBy("k").count().count()
    spark.stop()
    report(n == 7, "Spark can write and read Parquet")
except Exception as e:  # noqa: BLE001
    report(False, f"Spark end-to-end test: {type(e).__name__}: {str(e)[:400]}")

print("\nAll good - you can run the pipeline." if not problems else f"\n{problems} problem(s). Send a screenshot of this output.")
