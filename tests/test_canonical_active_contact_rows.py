"""Control-flow boundary checks; synthetic integer visits, no CPU physics."""

import ast
from types import SimpleNamespace

import pytest

from holomotion.src.evaluation.reference_tracking.canonical_gpu.row_kernels import (
    ActiveContactRows,
)

SOURCE = """
def kernel(nacon_in, values, visits):
    conid = wp.tid()
    if conid >= nacon_in[0]:
        return
    if values[conid] < 0:
        return
    visits.append(conid)
"""


@pytest.mark.parametrize("count", [0, 1, 5, 6, 9, -1])
def test_active_bound_preserves_order_and_original_early_returns(count):
    tree = ast.parse(SOURCE)
    transformed = ActiveContactRows(6).visit(tree)
    namespace = {"wp": SimpleNamespace(min=min)}
    exec(
        compile(
            ast.fix_missing_locations(transformed),
            "<contact-visit-test>",
            "exec",
        ),
        namespace,
    )
    values = [1, -1, 1, -1, 1, 1]
    visits = []
    namespace["kernel"]([count], values, visits)
    assert visits == [i for i in range(6) if i < count and values[i] >= 0]


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("conid >=", "conid >"),
        SOURCE.replace("conid = wp.tid()", "conid = 0"),
        SOURCE.replace(
            "    visits.append(conid)",
            "    for j in range(2):\n        if j == 0:\n            return",
        ),
    ],
)
def test_changed_upstream_guard_or_nested_return_is_rejected(source):
    with pytest.raises(ValueError):
        ActiveContactRows(6).visit(ast.parse(source))


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError, match="Invalid contact launch capacity"):
        ActiveContactRows(capacity)
