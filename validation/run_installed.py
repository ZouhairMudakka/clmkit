"""Build an sdist/wheel and test a non-editable install in a NumPy-only venv."""
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import tarfile
import venv

root = Path(__file__).resolve().parents[1]
artifacts = root / "validation-results"
artifacts.mkdir(exist_ok=True)
# Each build starts empty, so a previous wheel can never be tested by accident.
with tempfile.TemporaryDirectory(prefix="clmkit-build-") as build_folder:
    subprocess.run([sys.executable, "-m", "build", "--outdir", build_folder], cwd=root, check=True)
    wheels = list(Path(build_folder).glob("*.whl"))
    assert len(wheels) == 1, wheels
    sdists = list(Path(build_folder).glob("*.tar.gz"))
    assert len(sdists) == 1, sdists
    with tarfile.open(sdists[0]) as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        assert len(roots) == 1, roots
        paths = [PurePosixPath(*PurePosixPath(member.name).parts[1:]) for member in members if member.isfile()]
        allowed_root = {"README.md", "LICENSE", "CHANGELOG.md", "pyproject.toml", "PKG-INFO", ".gitignore"}
        unexpected = [str(path) for path in paths if not (
            str(path) in allowed_root or path.is_relative_to("src/clmkit") or path.is_relative_to("tests")
        ) or ".." in path.parts]
        assert not unexpected, f"Unexpected sdist files: {unexpected}"
        assert not any(member.issym() or member.islnk() for member in members), "sdist must not contain links"
    inventory = {"sdist_files": [str(path) for path in paths], "artifacts": {
        artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest() for artifact in [wheels[0], sdists[0]]
    }}
    (artifacts / "artifact-inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
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
