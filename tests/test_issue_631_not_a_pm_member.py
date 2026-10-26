"""#631: the plugin checkout must never become a pyproject-backed PM member.

Hermes' package manager (``pm/``, after v2026.9.24) makes any plugin checkout
with a ``pyproject.toml`` a uv workspace member. It adds a ``[project]`` name to
the member's ``pyproject.toml`` and keeps its other fields. A lint-only file has
no ``[project]`` table, so the result has a name and no ``version``, ``uv lock``
rejects it, and the update adds ``hermes-lcm-x`` to ``plugins.disabled``. Lint
settings therefore live in ``ruff.toml`` and the repo has no ``pyproject.toml``.

Fork: the manifest declares ``python_dependencies: [tiktoken]`` so every PM
generation installs the tokenizer (it is not a Hermes core dependency and was
dropped by each update). That makes the checkout a *manifest-only* member: PM
writes the member's own ``pyproject.toml`` (name, ``version = "0.0.0"``,
``package = false``) under ``plugin-deps/`` and copies nothing from the
checkout, so the #631 failure cannot occur. These tests pin that shape.

The staging test runs a real package manager only when ``LCM_REAL_HERMES_SRC``
names a Hermes tree with ``pm/workspace.py`` and ``LCM_REAL_HERMES_PYTHON`` names
its interpreter (the real-host variables of ``test_real_plugin_manager_*``).
"""

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_lint_settings_live_in_ruff_toml_not_pyproject():
    assert not (REPO_ROOT / "pyproject.toml").exists()

    config = tomllib.loads((REPO_ROOT / "ruff.toml").read_text(encoding="utf-8"))

    assert config["target-version"] == "py311"
    assert config["lint"]["select"] == ["E4", "E7", "E9", "F"]
    assert config["lint"]["ignore"] == ["E402"]


def test_checkout_is_a_manifest_only_member():
    # Mirrors PythonDeclaration.is_member in Hermes pm/plugin_declarations.py: not external
    # and (a pyproject.toml exists or the manifest declares python/pip dependencies).
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    top_level_keys = set(re.findall(r"^([A-Za-z_]+):", manifest, re.MULTILINE))

    assert "python_runtime" not in top_level_keys
    assert not (REPO_ROOT / "pyproject.toml").exists()
    assert top_level_keys & {"pip_dependencies", "python_dependencies"}
    block = manifest.split("python_dependencies:", 1)[1]
    specs = re.findall(r'^\s+-\s+"?([^"\n]+)"?\s*$', block.split("\n\n", 1)[0].split("provides_tools:", 1)[0],
                       re.MULTILINE)
    # Index requirements only (PM drops direct URLs), with an upper bound against Hermes' pins.
    assert specs == ["tiktoken>=0.7,<1"], specs


_STAGE_PROBE = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
try:
    import tomllib
    from pm import workspace
    from pm.plugin_declarations import read_python_declaration
    tree, checkout, home = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    declaration = read_python_declaration(checkout)
    result = {"is_member": declaration.is_member, "requirements": list(declaration.install_requirements)}
    root = home / "generation"
    identity = home / "plugins" / "hermes-lcm-x"
    workspace._generate_pyproject({identity: checkout}, root, source=tree)
    result["plugin_sources"] = (root / "plugin-sources").exists()
    members = sorted((root / "plugin-deps").iterdir())
    result["members"] = len(members)
    project = tomllib.loads((members[0] / "pyproject.toml").read_text())["project"]
    result["member_version"] = project.get("version")
    result["member_dependencies"] = project.get("dependencies")
except Exception as exc:  # a changed or missing package manager: skip, never fail
    result = {"skip": f"{type(exc).__name__}: {exc}"}
print(json.dumps(result))
"""


def test_installed_package_manager_stages_a_manifest_member(tmp_path):
    tree = Path(os.environ.get("LCM_REAL_HERMES_SRC") or "/nonexistent")
    python = Path(os.environ.get("LCM_REAL_HERMES_PYTHON") or "/nonexistent")
    if not (tree / "pm" / "workspace.py").is_file():
        pytest.skip("LCM_REAL_HERMES_SRC does not name a Hermes tree with the package manager (pm/)")
    if not python.is_file():
        pytest.skip("LCM_REAL_HERMES_PYTHON does not name an interpreter")
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "HERMES_HOME": str(tmp_path / "hermes-home"), "PATH": "/usr/bin:/bin"}
    # -B -I: import the Hermes tree read-only; every write goes under tmp_path.
    proc = subprocess.run(
        [str(python), "-B", "-I", "-c", _STAGE_PROBE, str(tree), str(REPO_ROOT), str(home)],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])

    assert result == {
        "is_member": True,
        "requirements": ["tiktoken>=0.7,<1"],
        "plugin_sources": False,
        "members": 1,
        "member_version": "0.0.0",
        "member_dependencies": ["tiktoken>=0.7,<1"],
    }
