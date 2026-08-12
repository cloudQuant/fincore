from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
GENERATOR = Path(__file__).parents[2] / "scripts" / "generate_compat_manifest.py"
FACTOR_PARTITIONS = {
    "style": ["momentum", "size", "value", "reversal_short_term", "volatility"],
    "sector": [
        "basic_materials",
        "consumer_cyclical",
        "financial_services",
        "real_estate",
        "consumer_defensive",
        "health_care",
        "utilities",
        "communication_services",
        "energy",
        "industrials",
        "technology",
    ],
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("compat_manifest_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _by_symbol(entries: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    return next(entry for entry in entries if entry["symbol"] == symbol)


def _parameter(entry: dict[str, Any], name: str) -> dict[str, Any]:
    return next(parameter for parameter in entry["parameters"] if parameter["name"] == name)


def _assert_portable_provenance(data: dict[str, Any]) -> None:
    serialized = json.dumps(data, sort_keys=True)
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    for source in data["source_files"]:
        assert not Path(source["path"]).is_absolute()
        assert len(source["sha256"]) == 64


def _assert_compatibility_levels(entry: dict[str, Any]) -> None:
    assert set(entry["compatibility"]) == {"C0", "C1", "C2", "C3", "C4"}


def _assert_reconstructable_signature(signature: str) -> None:
    ast.parse(f"def _f{signature}:\n    pass\n")
    namespace: dict[str, Any] = {}
    exec(f"def _f{signature}:\n    pass\n", {"__builtins__": {}}, namespace)
    assert str(inspect.signature(namespace["_f"])) == signature


def test_empyrical_manifest_is_pinned_and_complete() -> None:
    data = _load("empyrical-0.6.0-api.json")
    assert data["version"] == "0.6.0"
    assert data["commit"] == "74655e974ed2935563820c548c339731f1fe0621"
    assert len(data["public_symbols"]) == 54
    assert len(data["callables"]) == 49
    assert {"calmar_ratio", "beta", "perf_attrib", "DAILY"} <= set(data["public_symbols"])
    assert len(set(data["public_symbols"])) == 54
    assert all(entry["extraction"] == "static_ast_from_pinned_git_blob" for entry in data["symbols"])
    assert all(entry["reviewed"] is False for entry in data["symbols"])
    assert all(entry["signature"].startswith("(") for entry in data["callables"])
    for entry in data["callables"]:
        _assert_reconstructable_signature(entry["signature"])
    dynamic_review = [entry for entry in data["symbols"] if entry["needs_dynamic_review"]]
    assert len(dynamic_review) == 9
    assert all(entry["factory"].startswith("_create_") for entry in dynamic_review)
    for entry in data["symbols"]:
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_empyrical_canonical_signatures_use_runtime_defaults() -> None:
    data = _load("empyrical-0.6.0-api.json")
    calmar = _by_symbol(data["callables"], "calmar_ratio")
    beta = _by_symbol(data["callables"], "beta")
    assert calmar["signature"] == "(returns, period='daily', annualization=None)"
    assert _parameter(calmar, "period") == {
        "annotation": None,
        "default": "daily",
        "default_expression": "DAILY",
        "kind": "POSITIONAL_OR_KEYWORD",
        "name": "period",
        "required": False,
        "resolved": True,
    }
    assert beta["signature"] == "(returns, factor_returns, risk_free=0.0, out=None)"
    assert _parameter(beta, "risk_free")["default"] == 0.0
    assert _parameter(beta, "risk_free")["kind"] == "POSITIONAL_OR_KEYWORD"


def test_pyfolio_canonical_signatures_resolve_constants_and_expressions() -> None:
    profile = _load("pyfolio-0.9.6-api.json")["compatibility_profile"]
    capacity = profile["create_capacity_tear_sheet"]
    full = profile["create_full_tear_sheet"]
    assert capacity["signature"] == (
        "(returns, positions, transactions, market_data, "
        "liquidation_daily_vol_limit=0.2, trade_daily_vol_limit=0.05, "
        "last_n_days=126, days_to_liquidate_limit=1, estimate_intraday='infer', "
        "run_flask_app=False)"
    )
    assert _parameter(capacity, "last_n_days") == {
        "annotation": None,
        "default": 126,
        "default_expression": "utils.APPROX_BDAYS_PER_MONTH * 6",
        "kind": "POSITIONAL_OR_KEYWORD",
        "name": "last_n_days",
        "required": False,
        "resolved": True,
    }
    expected_full = (
        "(returns, positions=None, transactions=None, market_data=None, "
        "benchmark_rets=None, slippage=None, live_start_date=None, "
        "sector_mappings=None, bayesian=False, round_trips=False, "
        "estimate_intraday='infer', hide_positions=False, cone_std=(1.0, 1.5, 2.0), "
        "bootstrap=False, unadjusted_returns=None, style_factor_panel=None, "
        "sectors=None, caps=None, shares_held=None, volumes=None, percentile=None, "
        "turnover_denom='AGB', set_context=True, factor_returns=None, "
        "factor_loadings=None, pos_in_dollars=True, header_rows=None, "
        f"factor_partitions={FACTOR_PARTITIONS!r})"
    )
    assert full["signature"] == expected_full
    assert _parameter(full, "factor_partitions")["default"] == FACTOR_PARTITIONS
    assert _parameter(full, "factor_partitions")["resolved"] is True
    assert _parameter(full, "factor_partitions")["kind"] == "POSITIONAL_OR_KEYWORD"


def test_unresolved_defaults_are_not_labeled_signature_ready() -> None:
    generator = _load_generator()
    tree = ast.parse("def target(value=MISSING):\n    pass\n")
    function = tree.body[0]
    resolver = generator.StaticConstantResolver({"module": tree})
    parameters = generator._parameters(function, resolver, "module")
    assert parameters[0]["resolved"] is False
    assert parameters[0]["default"] is None
    assert parameters[0]["default_expression"] == "MISSING"
    assert "unknown name" in parameters[0]["unresolved_reason"]
    assert generator._canonical_signature(function, parameters) is None


def test_constant_resolver_rejects_huge_power_before_operator_runs(monkeypatch) -> None:
    generator = _load_generator()
    tree = ast.parse("VALUE = 2 ** 1000000")
    resolver = generator.StaticConstantResolver({"module": tree})

    def forbidden_power(_left: object, _right: object) -> object:
        pytest.fail("unsafe exponent reached operator.pow")

    monkeypatch.setitem(generator.SAFE_BINARY_OPERATORS, ast.Pow, forbidden_power)
    value, resolved = resolver.resolve(tree.body[0].value, "module")
    assert value is None
    assert resolved is False
    assert "exponent" in resolver.last_unresolved_reason


def test_constant_resolver_rejects_oversized_container_before_children(monkeypatch) -> None:
    generator = _load_generator()
    node = ast.List(
        elts=[ast.Constant(value=index) for index in range(generator.MAX_CONTAINER_ITEMS + 1)],
        ctx=ast.Load(),
    )
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    original = resolver._resolve
    visits = 0

    def counting_resolve(*args: Any, **kwargs: Any) -> Any:
        nonlocal visits
        visits += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(resolver, "_resolve", counting_resolve)
    assert resolver.resolve(node, "module") == (None, False)
    assert visits == 1
    assert "container" in resolver.last_unresolved_reason


@pytest.mark.parametrize(
    "node",
    [
        ast.Constant(value=b"bytes"),
        ast.Constant(value=1 + 2j),
        ast.Set(elts=[ast.Constant(value=1)]),
        ast.Constant(value=float("inf")),
        ast.Constant(value=float("nan")),
    ],
)
def test_constant_resolver_rejects_non_json_or_non_finite_values(node: ast.expr) -> None:
    generator = _load_generator()
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(node, "module") == (None, False)
    assert resolver.last_unresolved_reason


def test_constant_resolver_enforces_depth_and_visit_budgets() -> None:
    generator = _load_generator()
    deep: ast.expr = ast.Constant(value=1)
    for _ in range(generator.MAX_RESOLVE_DEPTH + 1):
        deep = ast.UnaryOp(op=ast.UAdd(), operand=deep)
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(deep, "module") == (None, False)
    assert "depth" in resolver.last_unresolved_reason

    nodes: list[ast.expr] = [ast.Constant(value=1) for _ in range(2048)]
    while len(nodes) > 1:
        nodes = [
            ast.BinOp(left=nodes[index], op=ast.Add(), right=nodes[index + 1]) for index in range(0, len(nodes), 2)
        ]
    resolver = generator.StaticConstantResolver({"module": ast.Module(body=[], type_ignores=[])})
    assert resolver.resolve(nodes[0], "module") == (None, False)
    assert "node visits" in resolver.last_unresolved_reason


def test_pyfolio_manifest_is_pinned() -> None:
    data = _load("pyfolio-0.9.6-api.json")
    assert data["version"] == "0.9.6"
    assert data["commit"] == "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a"
    assert "create_full_tear_sheet" in data["compatibility_profile"]
    assert "create_risk_tear_sheet" in data["compatibility_profile"]
    assert len(data["compatibility_profile"]) == 11
    for entry in data["compatibility_profile"].values():
        assert entry["public_path"].startswith("pyfolio.")
        assert isinstance(entry["parameters"], list)
        assert entry["signature"].startswith("(")
        _assert_reconstructable_signature(entry["signature"])
        _assert_compatibility_levels(entry)
    _assert_portable_provenance(data)


def test_flat_api_migration_manifest_is_explicit() -> None:
    data = _load("fincore-flat-api-migrations.json")
    assert data["fincore_version"] == "0.3.0"
    assert data["policy"] == "preserve-0.3.x-flat-api"
    assert data["entries"]
    for entry in data["entries"]:
        assert {
            "symbol",
            "current_target",
            "recommended_target",
            "deprecate_in",
            "remove_or_switch_in",
        } <= set(entry)


def test_generator_default_path_is_ast_only() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert {"empyrical", "pyfolio", "importlib", "inspect"}.isdisjoint(imported)
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "path"
        for node in ast.walk(tree)
    )
    assert _load("empyrical-0.6.0-api.json")["oracle_verification"] == {
        "reviewed": False,
        "status": "not_run",
    }


def test_static_exports_preserve_aliases_and_expand_star() -> None:
    generator = _load_generator()
    trees = {
        "__init__": ast.parse("from .direct import original as public_alias\nfrom .star import *\n"),
        "direct": ast.parse("def original():\n    pass\n"),
        "star": ast.parse("__all__ = ['chosen']\nfrom .impl import source_name as chosen\nhidden = 1\n"),
        "impl": ast.parse("def source_name():\n    pass\n"),
    }
    exports = generator._resolve_public_exports("__init__", trees)
    assert [(item.public_name, item.source_name, item.source_module) for item in exports] == [
        ("public_alias", "original", "direct"),
        ("chosen", "source_name", "impl"),
    ]


def test_static_star_without_all_uses_python_public_names() -> None:
    generator = _load_generator()
    trees = {
        "__init__": ast.parse("from .star import *\n"),
        "star": ast.parse("from .impl import source as alias\ndef local():\n    pass\n_hidden = 1\n"),
        "impl": ast.parse("def source():\n    pass\n"),
    }
    exports = generator._resolve_public_exports("__init__", trees)
    assert [(item.public_name, item.source_name, item.source_module) for item in exports] == [
        ("alias", "source", "impl"),
        ("local", "local", "star"),
    ]


def test_review_attestation_is_preserved_only_for_identical_evidence() -> None:
    generator = _load_generator()
    generated = {
        "project": "empyrical",
        "commit": "abc",
        "source_files": [{"path": "stats.py", "sha256": "1" * 64}],
        "oracle_verification": {"status": "not_run", "reviewed": False},
        "symbols": [{"symbol": "alpha", "signature": "(returns)", "reviewed": False}],
    }
    previous = json.loads(json.dumps(generated))
    previous["symbols"][0]["reviewed"] = True
    merged = generator._merge_review_attestations(generated, previous, "symbols")
    assert merged["symbols"][0]["reviewed"] is True
    drifted = json.loads(json.dumps(generated))
    drifted["symbols"][0]["signature"] = "(returns, factor=None)"
    invalidated = generator._merge_review_attestations(drifted, previous, "symbols")
    assert invalidated["symbols"][0]["reviewed"] is False


def test_pinned_source_reads_committed_blob_when_worktree_is_dirty(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "upstream"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    source = root / "package.py"
    source.write_text("VALUE = 'pinned'\n", encoding="utf-8")
    subprocess.run(["git", "add", "package.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    source.write_text("VALUE = 'dirty'\n", encoding="utf-8")
    pinned = generator.PinnedGitSource(root, commit)
    assert pinned.read_text("package.py") == "VALUE = 'pinned'\n"
    assert pinned.sha256("package.py") != generator.hashlib.sha256(source.read_bytes()).hexdigest()


def test_pinned_git_read_timeout_names_operation(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()
    source = object.__new__(generator.PinnedGitSource)
    source.root = tmp_path
    source.commit = "abc123"

    def time_out(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(generator.subprocess, "run", time_out)
    with pytest.raises(ValueError, match="read pinned Git blob.*timed out"):
        source.read_bytes("package.py")


def test_oracle_execution_timeout_names_operation(tmp_path: Path, monkeypatch) -> None:
    generator = _load_generator()

    def time_out(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(generator.subprocess, "run", time_out)
    with pytest.raises(ValueError, match="execute isolated oracle.*timed out"):
        generator._execute_oracle(Path(sys.executable), tmp_path, "samplepkg", ["target"], [])


def test_generator_subprocess_calls_are_centralized_and_bounded() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(calls) == 1
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_process")
    assert calls[0] in list(ast.walk(wrapper))
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert {"timeout", "env", "stdin"} <= keyword_names
    source = GENERATOR.read_text(encoding="utf-8")
    assert '"GIT_TERMINAL_PROMPT": "0"' in source
    assert "timeout=LOCAL_GIT_TIMEOUT_SECONDS" in source


def test_oracle_rejects_wrong_source_version(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "wrong"
    package = root / "samplepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "__version__ = '9.9.9'\ndef target(value=1):\n    return value\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "samplepkg/__init__.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(ValueError, match="oracle version"):
        generator._run_oracle(
            Path(sys.executable),
            root,
            "samplepkg",
            ["target"],
            expected_version="1.0.0",
            expected_commit=commit,
            expected_source_files=[{"path": "__init__.py", "sha256": "unused"}],
        )


def test_oracle_rejects_same_named_installed_package_outside_pin(tmp_path: Path) -> None:
    generator = _load_generator()
    root = tmp_path / "empty-pin"
    root.mkdir()
    (root / "README.md").write_text("no pytest package in this pin\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    with pytest.raises(ValueError, match="outside pinned root"):
        generator._run_oracle(
            Path(sys.executable),
            root,
            "pytest",
            ["main"],
            expected_version="installed-version-must-not-be-used",
            expected_commit=commit,
            expected_source_files=[],
        )


def test_full_generator_is_byte_idempotent_when_pinned_roots_are_available(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[4]
    empyrical_root = project_root / "empyrical"
    pyfolio_root = project_root / "pyfolio"
    if not (empyrical_root.is_dir() and pyfolio_root.is_dir()):
        pytest.skip("pinned sibling roots are not available; frozen fixtures remain CI input")
    command = [
        sys.executable,
        GENERATOR.as_posix(),
        "--empyrical-root",
        empyrical_root.as_posix(),
        "--pyfolio-root",
        pyfolio_root.as_posix(),
        "--output",
        tmp_path.as_posix(),
    ]
    subprocess.run(command, check=True)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    subprocess.run(command, check=True)
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    assert first == second
    assert set(first) == {
        "empyrical-0.6.0-api.json",
        "fincore-flat-api-migrations.json",
        "pyfolio-0.9.6-api.json",
    }
    for name, contents in first.items():
        assert contents == (FIXTURES / name).read_bytes()
