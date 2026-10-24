"""Pinned row-loop adapters; no generic shared-write classification."""

import ast


class RowContinues(ast.NodeTransformer):
    def visit_Return(self, node):
        if node.value is not None:
            raise ValueError("Unexpected value-returning physics kernel")
        return ast.copy_location(ast.Continue(), node)


class WorldRows(ast.NodeTransformer):
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def visit_FunctionDef(self, node):
        assign = next(
            s
            for s in node.body
            if isinstance(s, ast.Assign)
            and isinstance(s.targets[0], ast.Tuple)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute)
            and s.value.func.attr == "tid"
        )
        names = [s.id for s in assign.targets[0].elts]
        assert names[0] == "worldid" and len(names) == len(self.dimensions) + 1
        before = node.body[: node.body.index(assign)]
        after = node.body[node.body.index(assign) + 1 :]
        if any(
            isinstance(x, ast.Return)
            for s in after
            for loop in ast.walk(s)
            if isinstance(loop, (ast.For, ast.While))
            for x in ast.walk(loop)
        ):
            raise ValueError("Nested row return needs a separate adapter")
        after = [RowContinues().visit(s) for s in after]
        counters = [
            f"ordered_row_index_{i}" for i in range(len(self.dimensions))
        ]
        assignments = ast.parse(
            "worldid = ordered_world_index\n"
            + "".join(
                f"{name} = {counter}\n"
                for name, counter in zip(names[1:], counters)
            )
        ).body
        row_initialization = [
            s
            for s in before
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
        ]
        body = assignments + row_initialization + after
        for counter, dimension in reversed(
            list(zip(counters, self.dimensions))
        ):
            loop = ast.parse(
                f"for {counter} in range({dimension}):\n    pass\n"
            ).body[0]
            loop.body = body
            body = [loop]
        node.body = ast.parse("ordered_world_index = wp.tid()").body + body
        return ast.fix_missing_locations(node)


class BoundedRows(WorldRows):
    def visit_FunctionDef(self, node):
        expected = ast.dump(
            ast.parse("efcid >= nefc_in[worldid]", mode="eval").body
        )
        guarded = any(
            isinstance(s, ast.If)
            and ast.dump(s.test) == expected
            and len(s.body) == 1
            and isinstance(s.body[0], ast.Return)
            for s in node.body
        )
        transformed = super().visit_FunctionDef(node)
        if guarded:
            loops = [
                n
                for n in ast.walk(transformed)
                if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name)
                and n.target.id == "ordered_row_index_0"
            ]
            assert len(loops) == 1
            loops[0].iter.args[0] = ast.parse(
                f"wp.min({int(self.dimensions[0])},nefc_in[ordered_world_index])",
                mode="eval",
            ).body
        return ast.fix_missing_locations(transformed)


class PyramidalOnly(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "is_elliptic"
        ):
            node.value = ast.Constant(False)
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "is_elliptic":
            return node.orelse
        return self.generic_visit(node)


class ActiveContactRows(ast.NodeTransformer):
    """Keep one global contact order; skip only the original inactive tail."""

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("Invalid contact launch capacity")
        self.capacity = capacity

    def visit_FunctionDef(self, node):
        expected_tid = ast.dump(ast.parse("conid = wp.tid()").body[0])
        assignments = [s for s in node.body if ast.dump(s) == expected_tid]
        if len(assignments) != 1 or node.body.index(assignments[0]) != 0:
            raise ValueError("Unexpected contact kernel index initialization")
        after = node.body[1:]
        expected_guard = ast.dump(
            ast.parse("if conid >= nacon_in[0]:\n    return").body[0]
        )
        if not after or ast.dump(after[0]) != expected_guard:
            raise ValueError("Missing original contact capacity guard")
        if any(
            isinstance(x, ast.Return)
            for s in after
            for loop in ast.walk(s)
            if isinstance(loop, (ast.For, ast.While))
            for x in ast.walk(loop)
        ):
            raise ValueError("Nested contact return requires another adapter")
        loop = ast.parse(
            f"for conid in range(wp.min({self.capacity}, nacon_in[0])):\n    pass"
        ).body[0]
        loop.body = [RowContinues().visit(s) for s in after]
        node.body = [loop]
        return ast.fix_missing_locations(node)
