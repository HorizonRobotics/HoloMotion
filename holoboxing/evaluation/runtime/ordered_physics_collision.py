"""Stable per-pair collision workspaces for the pinned single arena.

Geometry calculations run independently. Contact append order remains the
upstream type, candidate-pair, and local-contact order. Integer prefix sums
assign broadphase rows without changing geometry or collision equations.
"""

import ast
import inspect
import types
import numpy as np
import warp as wp
from mujoco_warp._src import collision_driver as driver
from mujoco_warp._src.collision_primitive import write_contact

CONTACTS_PER_PAIR = 16


class ContactSlots(ast.NodeTransformer):
    def __init__(self, ngeom, stride):
        self.ngeom = ngeom
        self.stride = stride

    def visit_Assign(self, node):
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cid"
        ):
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "atomic_add"
            ):
                raise RuntimeError("Unexpected upstream contact allocation")
            return ast.parse(
                f"key = wp.min(geoms_in[0],geoms_in[1])*{self.ngeom}+wp.max(geoms_in[0],geoms_in[1])\nlocal_slot = wp.atomic_add(nacon_out,key,1)\nif local_slot >= {self.stride}:\n    return 0\ncid = key*{self.stride}+local_slot"
            ).body
        return self.generic_visit(node)


class CCDSlots(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "ccdid"
        ):
            if not (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "atomic_add"
            ):
                raise RuntimeError("Unexpected upstream CCD allocation")
            # Retain the original integer counter update. Typed candidates
            # already have the exact sequential scratch index.
            return [
                ast.copy_location(ast.Expr(value=node.value), node),
                ast.copy_location(
                    ast.Assign(
                        targets=node.targets,
                        value=ast.Name(id="collisionid", ctx=ast.Load()),
                    ),
                    node,
                ),
            ]
        return self.generic_visit(node)


def cloned(function, replacements):
    closure = tuple(
        types.CellType(replacements.get(name, cell.cell_contents))
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or ()
        )
    )
    new = types.FunctionType(
        function.__code__,
        dict(function.__globals__, **replacements),
        function.__name__,
        function.__defaults__,
        closure or None,
    )
    new.__annotations__ = function.__annotations__.copy()
    new.__module__ = function.__module__
    new.__qualname__ = function.__qualname__
    return new


@wp.kernel
def append_contacts(
    collision_count: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    ngeom: int,
    counts: wp.array(dtype=int),
    capacity: int,
    count: wp.array(dtype=int),
    stride: int,
    overflow: wp.array(dtype=int),
    src_dist: wp.array(dtype=float),
    dst_dist: wp.array(dtype=float),
    src_pos: wp.array(dtype=wp.vec3),
    dst_pos: wp.array(dtype=wp.vec3),
    src_frame: wp.array(dtype=wp.mat33),
    dst_frame: wp.array(dtype=wp.mat33),
    src_margin: wp.array(dtype=float),
    dst_margin: wp.array(dtype=float),
    src_friction: wp.array(dtype=wp.types.vector(5, float)),
    dst_friction: wp.array(dtype=wp.types.vector(5, float)),
    src_solref: wp.array(dtype=wp.vec2),
    dst_solref: wp.array(dtype=wp.vec2),
    src_solreffriction: wp.array(dtype=wp.vec2),
    dst_solreffriction: wp.array(dtype=wp.vec2),
    src_solimp: wp.array(dtype=wp.types.vector(5, float)),
    dst_solimp: wp.array(dtype=wp.types.vector(5, float)),
    src_dim: wp.array(dtype=int),
    dst_dim: wp.array(dtype=int),
    src_geom: wp.array(dtype=wp.vec2i),
    dst_geom: wp.array(dtype=wp.vec2i),
    src_world: wp.array(dtype=int),
    dst_world: wp.array(dtype=int),
    src_type: wp.array(dtype=int),
    dst_type: wp.array(dtype=int),
    src_collision: wp.array(dtype=int),
    dst_collision: wp.array(dtype=int),
):
    for i in range(wp.min(collision_count[0], pairs.shape[0])):
        pair = pairs[i]
        key = wp.min(pair[0], pair[1]) * ngeom + wp.max(pair[0], pair[1])
        if counts[key] > stride:
            wp.atomic_max(overflow, 0, 1)
        for local in range(wp.min(counts[key], stride)):
            src = key * stride + local
            dst = wp.atomic_add(count, 0, 1)
            if dst >= capacity:
                continue
            dst_dist[dst] = src_dist[src]
            dst_pos[dst] = src_pos[src]
            dst_frame[dst] = src_frame[src]
            dst_margin[dst] = src_margin[src]
            dst_friction[dst] = src_friction[src]
            dst_solref[dst] = src_solref[src]
            dst_solreffriction[dst] = src_solreffriction[src]
            dst_solimp[dst] = src_solimp[src]
            dst_dim[dst] = src_dim[src]
            dst_geom[dst] = src_geom[src]
            dst_world[dst] = src_world[src]
            dst_type[dst] = src_type[src]
            dst_collision[dst] = src_collision[src]


@wp.kernel
def select_type(
    types: wp.array(dtype=int),
    type1: int,
    type2: int,
    count: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    ids: wp.array(dtype=wp.vec2i),
    worlds: wp.array(dtype=int),
    outcount: wp.array(dtype=int),
    outpairs: wp.array(dtype=wp.vec2i),
    outids: wp.array(dtype=wp.vec2i),
    outworlds: wp.array(dtype=int),
    ccd_capacity: int,
    overflow: wp.array(dtype=int),
):
    n = int(0)
    for i in range(wp.min(count[0], pairs.shape[0])):
        pair = pairs[i]
        if types[pair[0]] == type1 and types[pair[1]] == type2:
            outpairs[n] = pair
            outids[n] = ids[i]
            outworlds[n] = worlds[i]
            n += 1
    outcount[0] = n
    if n > ccd_capacity:
        wp.atomic_max(overflow, 0, 1)


def broad_factory(opt, n_aabb, n_bound, n_margin):
    test = driver._broadphase_filter(opt, n_aabb, n_bound, n_margin)

    @wp.kernel(module="unique", enable_backward=False)
    def flags(
        aabb: wp.array3d(dtype=wp.vec3),
        bound: wp.array2d(dtype=float),
        margin: wp.array2d(dtype=float),
        pairs: wp.array(dtype=wp.vec2i),
        ids: wp.array(dtype=wp.vec2i),
        pos: wp.array2d(dtype=wp.vec3),
        mat: wp.array2d(dtype=wp.mat33),
        valid: wp.array(dtype=int),
    ):
        i = wp.tid()
        pair = pairs[i]
        valid[i] = int(
            test(aabb, bound, margin, pos, mat, pair[0], pair[1], 0)
            or ids[i][1] >= 0
        )

    return flags


@wp.kernel
def pack_parallel(
    geomtype: wp.array(dtype=int),
    pairs: wp.array(dtype=wp.vec2i),
    ids: wp.array(dtype=wp.vec2i),
    valid: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    capacity: int,
    count: wp.array(dtype=int),
    out: wp.array(dtype=wp.vec2i),
    outids: wp.array(dtype=wp.vec2i),
    worlds: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
):
    i = wp.tid()
    if i == 0:
        count[0] = prefix[prefix.shape[0] - 1] + valid[valid.shape[0] - 1]
        if count[0] > capacity:
            wp.atomic_max(overflow, 0, 1)
    if valid[i]:
        dst = prefix[i]
        if dst >= capacity:
            return
        pair = pairs[i]
        out[dst] = (
            wp.vec2i(pair[1], pair[0])
            if geomtype[pair[0]] > geomtype[pair[1]]
            else pair
        )
        outids[dst] = ids[i]
        worlds[dst] = 0


class CollisionPlan:
    def __init__(self, model, launch, overflow):
        from mujoco_warp._src.types import BroadphaseType

        self.model, self.launch, self.overflow = model, launch, overflow
        self.ngeom = model.geom_type.shape[0]
        if model.opt.broadphase != BroadphaseType.NXN:
            raise ValueError("Ordered arena requires NXN broadphase")
        geometry_types = model.geom_type.numpy()
        if np.isin(geometry_types, [1, 8]).any():
            raise ValueError(
                "Ordered arena does not support heightfields or SDF geometry"
            )
        pairs = np.sort(model.nxn_geom_pair_filtered.numpy(), axis=1)
        if not len(pairs) or len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(
                "Ordered arena requires unique geometry candidates"
            )
        device = model.geom_type.device
        self.valid = wp.empty(len(pairs), dtype=int, device=device)
        self.prefix = wp.empty_like(self.valid)
        self.flags = broad_factory(
            model.opt.broadphase_filter,
            model.geom_aabb.shape[0],
            model.geom_rbound.shape[0],
            model.geom_margin.shape[0],
        )
        self.kernels = {}
        self.buffers = self.counts = self.typed = None

    def broadphase(self, model, data, context):
        if model is not self.model or data.nworld != 1:
            raise ValueError("Ordered broadphase requires its single arena")
        self.launch(
            self.flags,
            dim=self.valid.size,
            inputs=[
                model.geom_aabb,
                model.geom_rbound,
                model.geom_margin,
                model.nxn_geom_pair_filtered,
                model.nxn_pairid_filtered,
                data.geom_xpos,
                data.geom_xmat,
                self.valid,
            ],
        )
        wp.utils.array_scan(self.valid, self.prefix, inclusive=False)
        self.launch(
            pack_parallel,
            dim=self.valid.size,
            inputs=[
                model.geom_type,
                model.nxn_geom_pair_filtered,
                model.nxn_pairid_filtered,
                self.valid,
                self.prefix,
                data.naconmax,
                data.ncollision,
                context.collision_pair,
                context.collision_pairid,
                context.collision_worldid,
                self.overflow,
            ],
        )

    @staticmethod
    def handles(kernel):
        return (
            "ccd_kernel_builder__locals__ccd_kernel" in kernel.key
            or "_primitive_narrowphase__locals__primitive_narrowphase"
            in kernel.key
        )

    def _kernel(self, original, primitive):
        if original in self.kernels:
            return self.kernels[original]
        module = wp.get_module("ordered_pairs_" + original.module.name)
        write = wp.Function(
            write_contact.func,
            key="pair_write",
            namespace="",
            module=module,
            code_transformers=[ContactSlots(self.ngeom, CONTACTS_PER_PAIR)],
        )
        if primitive:
            functions = original.adj.resolve_external_reference(
                "primitive_collisions_func"
            )
            replacements = []
            for function in functions:
                # The pinned wrappers call write_contact directly. Reject a
                # changed wrapper instead of retaining unordered nested writes.
                if (
                    function.func.__globals__.get("write_contact")
                    is not write_contact
                ):
                    raise RuntimeError(
                        "Unexpected upstream primitive contact wrapper"
                    )
                replacements.append(
                    wp.Function(
                        cloned(function.func, {"write_contact": write}),
                        key="pair_" + function.key,
                        namespace="",
                        module=module,
                    )
                )
            function = cloned(
                original.func,
                {"primitive_collisions_func": tuple(replacements)},
            )
            kernel = wp.Kernel(
                function,
                key="ordered_primitive",
                module=module,
                options={"enable_backward": False},
            )
        else:
            evaluation = original.adj.resolve_external_reference(
                "eval_ccd_write_contact"
            )
            evaluation = wp.Function(
                cloned(evaluation.func, {"write_contact": write}),
                key="pair_eval",
                namespace="",
                module=module,
            )
            function = cloned(
                original.func, {"eval_ccd_write_contact": evaluation}
            )
            kernel = wp.Kernel(
                function,
                key="ordered_ccd",
                module=module,
                options={"enable_backward": False},
                code_transformers=[CCDSlots()],
            )
        self.kernels[original] = kernel
        return kernel

    def run(self, kernel, *args, **kwargs):
        inputs, outputs = kwargs["inputs"].copy(), kwargs["outputs"]
        names = list(inspect.signature(kernel.func).parameters)
        fields = dict(zip(names, [*inputs, *outputs], strict=True))
        if len(outputs) != 14 or names[-1] != "nacon_out":
            raise RuntimeError("Unexpected upstream contact output layout")
        primitive = (
            "_primitive_narrowphase__locals__primitive_narrowphase"
            in kernel.key
        )
        device = outputs[0].device
        capacity = self.ngeom * self.ngeom * CONTACTS_PER_PAIR
        if self.buffers is None:
            self.buffers = [
                wp.empty(capacity, dtype=x.dtype, device=device)
                for x in outputs[:-1]
            ]
            self.counts = wp.empty(
                self.ngeom * self.ngeom, dtype=int, device=device
            )
            n = fields["collision_pair_in"].shape[0]
            self.typed = (
                wp.empty(1, dtype=int, device=device),
                wp.empty(n, dtype=wp.vec2i, device=device),
                wp.empty(n, dtype=wp.vec2i, device=device),
                wp.empty(n, dtype=int, device=device),
            )
        if not primitive:
            self.launch(
                select_type,
                dim=1,
                inputs=[
                    fields["geom_type"],
                    int(kernel.adj.resolve_external_reference("geomtype1")),
                    int(kernel.adj.resolve_external_reference("geomtype2")),
                    fields["ncollision_in"],
                    fields["collision_pair_in"],
                    fields["collision_pairid_in"],
                    fields["collision_worldid_in"],
                    *self.typed,
                    fields["naccdmax_in"],
                    self.overflow,
                ],
            )
            for name, value in zip(
                (
                    "ncollision_in",
                    "collision_pair_in",
                    "collision_pairid_in",
                    "collision_worldid_in",
                ),
                self.typed,
                strict=True,
            ):
                fields[name] = inputs[names.index(name)] = value
        self.counts.zero_()
        inputs[names.index("naconmax_in")] = capacity
        kwargs.update(
            inputs=inputs,
            outputs=[*self.buffers, self.counts],
            dim=fields["collision_pair_in"].shape[0],
            block_dim=1,
        )
        kwargs.pop("max_blocks", None)
        result = self.launch(self._kernel(kernel, primitive), *args, **kwargs)
        copy_inputs = [
            fields["ncollision_in"],
            fields["collision_pair_in"],
            self.ngeom,
            self.counts,
            fields["naconmax_in"],
            outputs[-1],
            CONTACTS_PER_PAIR,
            self.overflow,
        ]
        for source, destination in zip(
            self.buffers, outputs[:-1], strict=True
        ):
            copy_inputs.extend((source, destination))
        self.launch(append_contacts, dim=1, inputs=copy_inputs)
        return result
