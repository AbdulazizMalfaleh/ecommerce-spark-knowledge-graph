"""Windows helper: find Java 17/21 and the Hadoop helper files automatically.

Spark on Windows needs two things that are usually missing:
  * JAVA_HOME pointing at Java 17 or 21 (many PCs only have an old Java 8 on PATH)
  * HADOOP_HOME with bin\\winutils.exe and bin\\hadoop.dll, or writing files fails

configure() fixes both for the current Python process only (and the Spark/Java
process it starts), so no system environment variables need editing.
On Linux and macOS it does nothing.
"""
import glob
import os
import platform
import re
import urllib.request
from pathlib import Path

from src.config import PROJECT_ROOT

IS_WINDOWS = platform.system() == "Windows"

LOCAL_HADOOP = PROJECT_ROOT / "tools" / "hadoop"
HADOOP_FILES = ("winutils.exe", "hadoop.dll")
WINUTILS_URL = "https://raw.githubusercontent.com/cdarlint/winutils/master/{version}/bin/{file}"
WINUTILS_VERSIONS = ("hadoop-3.3.6", "hadoop-3.3.5")

JDK_ROOTS = (r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Java",
             r"C:\Program Files\Microsoft", r"C:\Program Files\Zulu", r"C:\Program Files\Amazon Corretto")
SUPPORTED_JAVA = (17, 21)


# ------------------------------ Java ------------------------------

def java_major(java_home: str | os.PathLike) -> int:
    """Read the version from the JDK's 'release' file, e.g. JAVA_VERSION="21.0.12" -> 21, "1.8.0_401" -> 8."""
    release = Path(java_home) / "release"
    if not release.exists():
        return 0
    m = re.search(r'JAVA_VERSION="(\d+)(?:\.(\d+))?', release.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        return 0
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 and m.group(2) else major


def find_java_home() -> str | None:
    """A Java 17/21 install: the current JAVA_HOME if it qualifies, otherwise search the usual folders."""
    current = os.environ.get("JAVA_HOME")
    if current and java_major(current) in SUPPORTED_JAVA:
        return current
    candidates = []
    for root in JDK_ROOTS:
        candidates += glob.glob(os.path.join(root, "jdk*"))
    good = [c for c in candidates if (Path(c) / "bin" / "java.exe").exists() and java_major(c) in SUPPORTED_JAVA]
    return max(good, key=java_major) if good else None       # prefer 21 over 17


# ------------------------------ Hadoop ------------------------------

def _has_hadoop_files(home: str | os.PathLike) -> bool:
    return all((Path(home) / "bin" / f).exists() for f in HADOOP_FILES)


def find_hadoop_home() -> str | None:
    for home in (os.environ.get("HADOOP_HOME"), r"C:\hadoop", str(LOCAL_HADOOP)):
        if home and _has_hadoop_files(home):
            return home
    return None


def download_hadoop_files(target: Path | None = None) -> str:
    """Download winutils.exe + hadoop.dll into <project>/tools/hadoop/bin."""
    target = target or LOCAL_HADOOP
    bin_dir = target / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for version in WINUTILS_VERSIONS:
        try:
            for f in HADOOP_FILES:
                url = WINUTILS_URL.format(version=version, file=f)
                print(f"  downloading {f} ({version}) ...")
                urllib.request.urlretrieve(url, bin_dir / f)
            return str(target)
        except Exception as e:  # noqa: BLE001  (try the next version)
            last_error = e
    raise RuntimeError(f"Could not download the Hadoop helper files: {last_error}")


# ------------------------------ apply ------------------------------

def _prepend_path(directory: str) -> None:
    path = os.environ.get("PATH", "")
    if directory.lower() not in path.lower().split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + path


def configure(download: bool = True) -> dict:
    """Set JAVA_HOME / HADOOP_HOME / PATH for this process. Returns what was used (empty on non-Windows)."""
    if not IS_WINDOWS:
        return {}

    used = {}
    java_home = find_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        _prepend_path(str(Path(java_home) / "bin"))       # ahead of any old Java 8 on PATH
        used["JAVA_HOME"] = java_home

    hadoop_home = find_hadoop_home()
    if not hadoop_home and download:
        hadoop_home = download_hadoop_files()
    if hadoop_home:
        os.environ["HADOOP_HOME"] = hadoop_home
        _prepend_path(str(Path(hadoop_home) / "bin"))
        used["HADOOP_HOME"] = hadoop_home
    return used
