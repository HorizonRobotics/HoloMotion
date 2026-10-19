"""Independent physics rows with deterministic allocation and scalar sums."""

import ast
import inspect
import numpy as np
import warp as wp
from . import ordered_physics_rows as rows


@wp.kernel
def allocate_rows(
    count: wp.array(dtype=int),
    types: wp.array(dtype=int),
    dims: wp.array(dtype=int),
    dist: wp.array(dtype=float),
    margin: wp.array(dtype=float),
    world: wp.array(dtype=int),
    capacity: int,
    nefc: wp.array(dtype=int),
    address: wp.array2d(dtype=int),
):
    for con in range(wp.min(count[0], address.shape[0])):
        if not types[con] & 1 or dist[con] - margin[con] >= 0.0:
            continue
        dim = dims[con]
        for edge in range(address.shape[1]):
            if (dim == 1 and edge > 0) or (dim > 1 and edge >= 2 * (dim - 1)):
                continue
            row = wp.atomic_add(nefc, world[con], 1)
            address[con, edge] = row if row < capacity else -1


class AllocatedContact(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "efcid"
        ):
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "atomic_add"
            ):
                raise RuntimeError(
                    "Unexpected upstream contact row allocation"
                )
            node.value = (
                ast.parse("contact_efc_address_out[conid,dimid]").body[0].value
            )
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if isinstance(node.left, ast.Name) and node.left.id == "efcid":
            if not isinstance(node.ops[0], ast.GtE):
                raise RuntimeError(
                    "Unexpected upstream contact row capacity check"
                )
            return ast.parse("efcid < 0").body[0].value
        return self.generic_visit(node)


@wp.kernel
def friction_slots(
    friction: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
):
    for dof in range(friction.shape[1]):
        if friction[0, dof] > 0.0:
            mapping[dof + 1] = wp.atomic_add(nefc, 0, 1)


class FrictionSlots(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "efcid"
        ):
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "atomic_add"
            ):
                raise RuntimeError(
                    "Unexpected upstream friction row allocation"
                )
            node.value = ast.parse("nefc_out[dofid+1]").body[0].value
        return self.generic_visit(node)


@wp.kernel
def ordered_sum(
    check: bool,
    done: wp.array(dtype=bool),
    values: wp.array(dtype=float),
    out: wp.array(dtype=float),
):
    if check:
        if done[0]:
            return
    for i in range(values.shape[0]):
        wp.atomic_add(out, 0, values[i])


class Terms(ast.NodeTransformer):
    def visit_Expr(self, node):
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "atomic_add"
        ):
            if not (
                len(value.args) == 3
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id
                in ("ctx_search_dot_out", "ctx_grad_dot_out")
            ):
                raise RuntimeError(
                    "Unexpected upstream solver scalar reduction"
                )
            return ast.copy_location(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=value.args[0],
                            slice=ast.Name(id="dofid", ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=value.args[2],
                ),
                node,
            )
        return self.generic_visit(node)


class CostTerms(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "atomic_add"
        ):
            if not (
                len(call.args) == 3
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "ctx_cost_out"
            ):
                raise RuntimeError(
                    "Unexpected upstream constraint cost reduction"
                )
            return ast.copy_location(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=call.args[0],
                            slice=ast.Name(id="efcid", ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=call.args[2],
                ),
                node,
            )
        return self.generic_visit(node)


@wp.kernel
def accumulate(
    nefc: wp.array(dtype=int),
    done: wp.array(dtype=bool),
    terms: wp.array(dtype=float),
    cost: wp.array(dtype=float),
):
    if done[0]:
        return
    for i in range(wp.min(nefc[0], terms.shape[0])):
        if terms[i] != 0.0:
            wp.atomic_add(cost, 0, terms[i])


class Collapse(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        at = next(
            i
            for i, x in enumerate(node.body)
            if isinstance(x, ast.Assign)
            and isinstance(x.targets[0], ast.Tuple)
        )
        if [n.id for n in node.body[at].targets[0].elts] != [
            "worldid",
            "nodeid",
        ]:
            raise RuntimeError("Unexpected upstream body-tree work assignment")
        after = [rows.RowReturns().visit(x) for x in node.body[at + 1 :]]
        loop = ast.parse(
            "worldid=0\nfor nodeid in range(body_tree_.shape[0]):\n    pass"
        ).body
        loop[-1].body = after
        node.body = node.body[:at] + loop
        return ast.fix_missing_locations(node)


class RowPlan:
    INDEPENDENT = frozenset(
        (
            "contact_force_kernel",
            "update_gradient_set_h_qM_lower_sparse",
            "_qM_sparse",
            "_apply_ft",
            "_cinert",
            "linesearch_qacc_ma",
            "linesearch_jaref",
        )
    )
    SCALAR = frozenset(
        ("solve_init_search", "solve_search_update", "update_gradient_grad")
    )
    TREES = frozenset(
        ("_subtree_com_acc", "_crb_accumulate", "_cfrc_backward")
    )
    ROWS = frozenset(
        (
            "_efc_contact_pyramidal",
            "_efc_friction_dof",
            "update_constraint_efc",
        )
    )

    def __init__(self, model, launch):
        self.model, self.launch = model, launch
        self.device = model.body_parentid.device
        self.kernels, self.buffers = {}, {}
        self.never_done = wp.zeros(1, dtype=bool, device=self.device)
        self.tree = wp.array(
            np.concatenate([a.numpy() for a in reversed(model.body_tree)]),
            dtype=int,
            device=self.device,
        )
        pairs = np.stack(
            (model.qM_fullm_i.numpy(), model.qM_fullm_j.numpy()), axis=-1
        )
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(
                "Sparse mass entries must own unique Hessian cells"
            )

    def handles(self, kernel):
        return (
            kernel.key
            in self.INDEPENDENT | self.SCALAR | self.TREES | self.ROWS
        )

    def _buffer(self, name, size, dtype=float):
        key = (name, size)
        if key not in self.buffers:
            self.buffers[key] = wp.empty(size, dtype=dtype, device=self.device)
        return self.buffers[key]

    def _kernel(self, original, transformers):
        if original not in self.kernels:
            name = "ordered_rows_" + original.key
            self.kernels[original] = wp.Kernel(
                original.func,
                key=name,
                module=wp.get_module(name + "_" + original.module.name),
                options={"enable_backward": False},
                code_transformers=transformers,
            )
        return self.kernels[original]

    def run(self, kernel, *args, **kwargs):
        if kernel.key in self.INDEPENDENT:
            # One output per body, DOF, contact, or unique sparse matrix cell.
            # Local vector mutations and calls to helpers need no serialization.
            return self.launch(kernel, *args, **kwargs)
        names = list(inspect.signature(kernel.func).parameters)
        inputs, outputs = kwargs["inputs"].copy(), kwargs["outputs"].copy()
        fields = dict(zip(names, [*inputs, *outputs], strict=True))
        if kernel.key == "_efc_contact_pyramidal":
            self.launch(
                allocate_rows,
                dim=1,
                inputs=[
                    fields[k]
                    for k in (
                        "nacon_in",
                        "type_in",
                        "condim_in",
                        "dist_in",
                        "includemargin_in",
                        "worldid_in",
                        "njmax_in",
                        "nefc_out",
                        "contact_efc_address_out",
                    )
                ],
            )
            kwargs["dim"] = fields["contact_efc_address_out"].shape
            return self.launch(
                self._kernel(kernel, [AllocatedContact()]), *args, **kwargs
            )
        if kernel.key == "_efc_friction_dof":
            mapping = self._buffer("friction", self.model.nv + 1, int)
            self.launch(
                friction_slots,
                dim=1,
                inputs=[
                    fields["dof_frictionloss"],
                    fields["nefc_out"],
                    mapping,
                ],
            )
            if names[len(inputs) + 1] != "nefc_out":
                raise RuntimeError("Unexpected friction row output layout")
            outputs[1] = mapping
            kwargs["outputs"] = outputs
            return self.launch(
                self._kernel(kernel, [FrictionSlots()]), *args, **kwargs
            )
        if kernel.key == "update_constraint_efc":
            cost = fields["ctx_cost_out"]
            terms = self._buffer("cost", fields["efc_D_in"].shape[1])
            terms.zero_()
            if names[-1] != "ctx_cost_out":
                raise RuntimeError("Unexpected constraint cost output layout")
            kwargs.update(outputs=[*outputs[:-1], terms], dim=(1, terms.size))
            result = self.launch(
                self._kernel(kernel, [rows.DropElliptic(), CostTerms()]),
                *args,
                **kwargs,
            )
            self.launch(
                accumulate,
                dim=1,
                inputs=[fields["nefc_in"], fields["ctx_done_in"], terms, cost],
            )
            return result
        if kernel.key in self.SCALAR:
            if names[-1] not in ("ctx_grad_dot_out", "ctx_search_dot_out"):
                raise RuntimeError("Unexpected solver scalar output layout")
            terms = self._buffer("scalar", self.model.nv)
            kwargs["outputs"] = [*outputs[:-1], terms]
            result = self.launch(
                self._kernel(kernel, [Terms()]), *args, **kwargs
            )
            done = fields.get("ctx_done_in")
            self.launch(
                ordered_sum,
                dim=1,
                inputs=[
                    done is not None,
                    done if done is not None else self.never_done,
                    terms,
                    outputs[-1],
                ],
            )
            return result
        if kernel.key in self.TREES:
            tree = fields["body_tree_"]
            if tree.ptr not in {a.ptr for a in self.model.body_tree}:
                raise RuntimeError("Unexpected ordered body-tree traversal")
            if tree.ptr != self.model.body_tree[-1].ptr:
                return None
            if inputs[-1].ptr != tree.ptr:
                raise RuntimeError("Unexpected body-tree argument layout")
            inputs[-1] = self.tree
            kwargs.update(inputs=inputs, dim=1, block_dim=1, max_blocks=1)
            return self.launch(
                self._kernel(kernel, [Collapse()]), *args, **kwargs
            )
        raise RuntimeError("Unsupported ordered physics row kernel")
