"""Exact contact ordering algebra only; these tests run no physics."""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    ROOT
    / "holomotion/src/evaluation/reference_tracking/canonical_gpu/driver.py"
)
module = ast.parse(DRIVER.read_text())
cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
method = next(
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "sort_contacts"
)
namespace = dict(
    torch=torch,
    dataclasses=dataclasses,
    wp=SimpleNamespace(to_torch=lambda x: x),
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        str(DRIVER),
        "exec",
    ),
    namespace,
)
sort_contacts = namespace["sort_contacts"]
Contact = dataclasses.make_dataclass(
    "Contact",
    [
        (x, object)
        for x in ("geomcollisionid", "pos", "dist", "geom", "worldid", "rowid")
    ],
)


def reference_order(arrays, count):
    order = torch.arange(
        len(arrays["worldid"]), device=arrays["worldid"].device
    )
    valid = order < count
    keys = [
        arrays["geomcollisionid"],
        *[arrays["pos"][:, k] for k in (2, 1, 0)],
        arrays["dist"],
        arrays["geom"][:, 1],
        arrays["geom"][:, 0],
        arrays["worldid"],
    ]
    for key in keys:
        safe = torch.where(valid, key, torch.zeros_like(key))
        order = order[torch.argsort(safe[order], stable=True)]
    return order[torch.argsort((~valid[order]).to(torch.int32), stable=True)]


@pytest.mark.parametrize("count", [0, 1, 71, 285, 5119, 5120, 5121, -1])
def test_exact_original_order_at_boundaries(count):
    torch.manual_seed(16)
    n = 5120
    arrays = dict(
        worldid=torch.randint(40, (n,), dtype=torch.int32),
        geom=torch.randint(72, (n, 2), dtype=torch.int32),
        geomcollisionid=torch.randint(3, (n,), dtype=torch.int32),
        pos=torch.randint(3, (n, 3)).float(),
        dist=torch.randint(3, (n,)).float(),
        rowid=torch.arange(n),
    )
    # Identical tuples, signed zero, infinity and NaN exercise exact float
    # comparison behavior without replacing any original float sort pass.
    special = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")]
    )
    arrays["dist"][:5] = special
    arrays["worldid"][:8] = 0
    arrays["geom"][:8] = 0
    valid_count = max(0, min(count, n))
    arrays["worldid"][valid_count:] = torch.iinfo(torch.int32).min
    arrays["geom"][valid_count:] = torch.iinfo(torch.int32).max
    order = reference_order(arrays, count)
    expected = {
        k: v[order].contiguous().view(torch.uint8).clone()
        for k, v in arrays.items()
    }
    data = SimpleNamespace(
        contact=Contact(**arrays), nworld=40, nacon=torch.tensor([count])
    )
    sort_contacts(None, data)
    for key, bits in expected.items():
        assert torch.equal(
            getattr(data.contact, key).contiguous().view(torch.uint8), bits
        ), key


def test_exhaustive_integer_domain_strict_embedding():
    triples = torch.cartesian_prod(
        torch.arange(40), torch.arange(72), torch.arange(72)
    ).to(torch.int32)
    keys = (triples[:, 0] * 72 + triples[:, 1]) * 72 + triples[:, 2]
    assert torch.equal(keys, torch.arange(40 * 72 * 72, dtype=torch.int32))
    assert keys[-1] < 40 * 72 * 72


@pytest.mark.parametrize(
    "nworld", [0, -1, torch.iinfo(torch.int32).max // (72 * 72) + 1]
)
def test_world_range_rejects_overflow(nworld):
    arrays = dict(
        worldid=torch.zeros(1, dtype=torch.int32),
        geom=torch.zeros((1, 2), dtype=torch.int32),
        geomcollisionid=torch.zeros(1, dtype=torch.int32),
        pos=torch.zeros((1, 3)),
        dist=torch.zeros(1),
        rowid=torch.arange(1),
    )
    with pytest.raises(RuntimeError, match="world range"):
        sort_contacts(
            None,
            SimpleNamespace(
                contact=Contact(**arrays),
                nworld=nworld,
                nacon=torch.tensor([0]),
            ),
        )
