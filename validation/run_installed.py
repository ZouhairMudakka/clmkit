"""Build an sdist/wheel and test a non-editable install in a NumPy-only venv."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

root = Path(__file__).resolve().parents[1]
artifacts = root / "validation-results"
artifacts.mkdir(exist_ok=True)
subprocess.run([sys.executable, "-m", "build", "--outdir", str(artifacts)], cwd=root, check=True)
wheel = next(artifacts.glob("*.whl"))
with tempfile.TemporaryDirectory(prefix="clmkit-installed-") as folder:
    work = Path(folder)
    venv.EnvBuilder(with_pip=True).create(work / "env")
    python = work / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", str(wheel), "pytest"], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    subprocess.run([str(python), "-I", "-c", "import importlib.util, sys, clmkit; "
                    "assert importlib.util.find_spec('torch') is None; "
                    "assert importlib.util.find_spec('transformers') is None; "
                    "assert 'site-packages' in clmkit.__file__; print(clmkit.__file__)"], cwd=work, check=True)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (artifacts / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
    shutil.copytree(root / "tests", work / "tests")
    shutil.copy2(root / "pyproject.toml", work / "pyproject.toml")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([str(python), "-I", "-m", "pytest", "-q", "-m", "not slow",
                             "--import-mode=importlib", "--basetemp", str(work / "tmp"),
                             "--junitxml", str(artifacts / "installed-junit.xml"), "tests"],
                            cwd=work, env=env)
    (artifacts / "installed-result.json").write_text(json.dumps({"exit_code": result.returncode,
                                                                 "python": sys.version}), encoding="utf-8")
    raise SystemExit(result.returncode)
