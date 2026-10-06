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
# Each build starts empty, so a previous wheel can never be tested by accident.
with tempfile.TemporaryDirectory(prefix="clmkit-build-") as build_folder:
    subprocess.run([sys.executable, "-m", "build", "--outdir", build_folder], cwd=root, check=True)
    wheels = list(Path(build_folder).glob("*.whl"))
    assert len(wheels) == 1, wheels
    wheel = artifacts / wheels[0].name
    for artifact in Path(build_folder).iterdir():
        shutil.copy2(artifact, artifacts / artifact.name)
with tempfile.TemporaryDirectory(prefix="clmkit-installed-") as folder:
    work = Path(folder)
    venv.EnvBuilder(with_pip=True).create(work / "env")
    python = work / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTHONPATH", None)
    subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-cache-dir", str(wheel), "pytest"], check=True, env=env)
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True, env=env)
    subprocess.run([str(python), "-I", "-c", "import importlib.util, sys, clmkit; "
                    "assert importlib.util.find_spec('torch') is None; "
                    "assert importlib.util.find_spec('transformers') is None; "
                    "assert 'site-packages' in clmkit.__file__; print(clmkit.__file__)"], cwd=work, check=True)
    freeze = subprocess.check_output([str(python), "-I", "-m", "pip", "freeze"], text=True, env=env)
    (artifacts / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    quickstart = readme.split("## Quickstart", 1)[1].split("```python", 1)[1].split("```", 1)[0]
    (work / "quickstart.py").write_text(quickstart, encoding="utf-8")
    quickstart_output = subprocess.check_output([str(python), "-I", str(work / "quickstart.py")], cwd=work, env=env, text=True)
    (artifacts / "quickstart-output.txt").write_text(quickstart_output, encoding="utf-8")
    print(quickstart_output)
    subprocess.run([str(python), "-I", str(root / "examples" / "01_quickstart.py")], cwd=work, env=env, check=True)
    shutil.copytree(root / "tests", work / "tests")
    shutil.copy2(root / "pyproject.toml", work / "pyproject.toml")
    result = subprocess.run([str(python), "-I", "-m", "pytest", "-q", "-m", "not slow",
                             "--import-mode=importlib", "--basetemp", str(work / "tmp"),
                             "--junitxml", str(artifacts / "installed-junit.xml"), "tests"],
                            cwd=work, env=env)
    (artifacts / "installed-result.json").write_text(json.dumps({"exit_code": result.returncode,
                                                                 "python": sys.version}), encoding="utf-8")
    raise SystemExit(result.returncode)
