"""Ordered active-row variants of pinned MuJoCo Warp kernels.

Warp's AST transformer retains upstream equations and argument layouts. It
changes the work assignment only: a single lane visits allocated rows in order,
instead of scanning every capacity slot. Elliptic rows are excluded explicitly;
the caller validates pyramidal model options before using this specialization.
"""

import ast
import warp as wp


class DropElliptic(ast.NodeTransformer):
    def visit_If(self, node):
        if (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.comparators[0], ast.Attribute)
            and node.test.comparators[0].attr == "CONTACT_ELLIPTIC"
        ):
            if not isinstance(node.test.ops[0], ast.NotEq):
                raise RuntimeError("Unexpected upstream friction-cone branch")
            return node.body
        return self.generic_visit(node)


class RowReturns(ast.NodeTransformer):
    def visit_Return(self, node):
        assert node.value is None
        return ast.copy_location(ast.Continue(), node)


class Compact(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = DropElliptic().visit(node)
        assign = next(
            x
            for x in node.body
            if isinstance(x, ast.Assign)
            and isinstance(x.targets[0], ast.Tuple)
        )
        assert [x.id for x in assign.targets[0].elts] == ["worldid", "efcid"]
        before = node.body[: node.body.index(assign)]
        after = node.body[node.body.index(assign) + 1 :]
        assert not any(
            isinstance(x, (ast.For, ast.While))
            for statement in after
            for x in ast.walk(statement)
        ), "Unexpected nested control flow"
        after = [RowReturns().visit(x) for x in after]
        template = ast.parse(
            "worldid = wp.tid()\nif ctx_done_in[worldid]:\n    return\nfor efcid in range(nefc_in[worldid]):\n    pass\n"
        )
        template.body[-1].body = after
        node.body = before + template.body
        return ast.fix_missing_locations(node)


_kernel = None
_contact_kernel = None
_ccd_kernels = {}


class CompactCCD(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        assign = next(
            x
            for x in node.body
            if isinstance(x, ast.Assign)
            and isinstance(x.targets[0], ast.Name)
            and isinstance(x.value, ast.Call)
            and isinstance(x.value.func, ast.Attribute)
            and x.value.func.attr == "tid"
        )
        assert assign.targets[0].id == "collisionid"
        before = node.body[: node.body.index(assign)]
        after = node.body[node.body.index(assign) + 1 :]
        assert not any(
            isinstance(x, (ast.For, ast.While))
            for statement in after
            for x in ast.walk(statement)
        )
        after = [RowReturns().visit(x) for x in after]
        template = ast.parse(
            "for collisionid in range(wp.min(ncollision_in[0],collision_pair_in.shape[0])):\n    pass\n"
        )
        template.body[0].body = after
        node.body = before + template.body
        return ast.fix_missing_locations(node)


def ccd_kernel(original):
    if original not in _ccd_kernels:
        module = wp.get_module(original.module.name + "_ordered_active")
        _ccd_kernels[original] = wp.Kernel(
            original.func,
            key=original.key + "_ordered_active",
            module=module,
            options={"enable_backward": False},
            code_transformers=[CompactCCD()],
        )
    return _ccd_kernels[original]


class CompactContact(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        assign = next(
            x
            for x in node.body
            if isinstance(x, ast.Assign)
            and isinstance(x.targets[0], ast.Tuple)
        )
        assert [x.id for x in assign.targets[0].elts] == ["conid", "dimid"]
        before = node.body[: node.body.index(assign)]
        after = node.body[node.body.index(assign) + 1 :]
        assert not any(
            isinstance(x, ast.Return)
            for statement in after
            for loop in ast.walk(statement)
            if isinstance(loop, (ast.For, ast.While))
            for x in ast.walk(loop)
        ), "Nested return needs separate treatment"
        after = [RowReturns().visit(x) for x in after]
        template = ast.parse(
            "for conid in range(wp.min(nacon_in[0], contact_efc_address_out.shape[0])):\n    for dimid in range(contact_efc_address_out.shape[1]):\n        pass\n"
        )
        template.body[0].body[0].body = after
        node.body = before + template.body
        return ast.fix_missing_locations(node)


def contact_kernel(original):
    global _contact_kernel
    if _contact_kernel is None:
        _contact_kernel = wp.Kernel(
            original.func,
            key="ordered_active_contact_rows_v1",
            module=wp.get_module("ordered_active_contact_rows_v1"),
            options={"enable_backward": False},
            code_transformers=[CompactContact()],
        )
    return _contact_kernel


def kernel(original):
    global _kernel
    if _kernel is None:
        _kernel = wp.Kernel(
            original.func,
            key="ordered_active_constraint_rows_v1",
            module=wp.get_module("ordered_active_constraint_rows_v1"),
            options={"enable_backward": False},
            code_transformers=[Compact()],
        )
    return _kernel
