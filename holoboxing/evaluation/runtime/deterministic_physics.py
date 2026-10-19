"""Fixed GPU reduction and allocation order for a single competition arena.

MuJoCo Warp's floating atomic reductions and contact allocation are otherwise
schedule dependent. Only kernels with shared writes are serialized. Cooperative
tile kernels retain their lanes; independent rows use the upstream gather path.
Launch changes apply inside ``ordered_physics`` only, including CUDA capture.
This adapter deliberately targets the versions pinned in the evaluation image.
"""

import ast
from contextlib import contextmanager
from functools import lru_cache
from importlib.metadata import version
from threading import Lock, local

import warp as wp

from . import ordered_physics_rows

ORDERING_VERSION = "single_arena_ordered_gpu_v2"

_state = local()
_installed = False
_install_lock = Lock()
_launch = wp.launch
_launch_tiled = wp.launch_tiled
_factor = _solve = _broadphase = None


class OrderedArena:
    """Own all fixed-address scratch buffers for one immutable GPU model."""

    def __init__(self, model):
        from .ordered_physics_linear import SparsePlan
        from .ordered_physics_collision import CollisionPlan
        from .ordered_physics_work import RowPlan

        if (model.nq, model.nv, model.nu) != (
            72,
            70,
            58,
        ) or not model.is_sparse:
            raise ValueError(
                "Ordered arena requires the sparse two-fighter model"
            )
        if not model.M_rowadr.device.is_cuda:
            raise ValueError("Ordered arena requires CUDA")
        self.model = model
        self.overflow = wp.zeros(1, dtype=int, device=model.M_rowadr.device)
        self.linear = SparsePlan(model, _launch)
        self.collision = CollisionPlan(model, _launch, self.overflow)
        self.rows = RowPlan(model, _launch)


def _arena():
    return (
        getattr(_state, "arena", None)
        if getattr(_state, "active", False)
        else None
    )


def _ordered_factor(model, data, matrix, factor, diagonal):
    arena = _arena()
    if arena is not None:
        return arena.linear.factor(model, data, matrix, factor, diagonal)
    return _factor(model, data, matrix, factor, diagonal)


def _ordered_solve(model, data, factor, diagonal, solution, right_hand_side):
    arena = _arena()
    if arena is not None:
        return arena.linear.solve(
            model, data, factor, diagonal, solution, right_hand_side
        )
    return _solve(model, data, factor, diagonal, solution, right_hand_side)


def _ordered_broadphase(model, data, context):
    arena = _arena()
    if arena is not None:
        return arena.collision.broadphase(model, data, context)
    return _broadphase(model, data, context)


@lru_cache(maxsize=None)
def _shared_writes(function):
    """Conservatively include indirect calls and implicit array += atomics."""
    return _visit_shared_writes(function, set())


def _visit_shared_writes(function, visiting):
    if id(function) in visiting:
        return False
    visiting.add(id(function))
    try:
        tree = function.adj.tree
        if any(
            (
                isinstance(node, (ast.Attribute, ast.Name))
                and "atomic_" in getattr(node, "attr", getattr(node, "id", ""))
            )
            or (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Subscript)
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Call, ast.Subscript))
            )
            for node in ast.walk(tree)
        ):
            return True
        _, _, called = function.adj.get_references()
        return any(_visit_shared_writes(child, visiting) for child in called)
    except Exception:
        # An uninspectable kernel cannot safely take the unordered path.
        return True
    finally:
        visiting.remove(id(function))


def _physics(kernel):
    return getattr(kernel.func, "__module__", "").startswith("mujoco_warp")


def _dimension(args, kwargs):
    return kwargs.get("dim", args[0] if args else None)


def _replace_dimension(args, kwargs, dimension):
    if "dim" in kwargs:
        kwargs["dim"] = dimension
    else:
        args = (dimension, *args[1:])
    return args


def _ordered_launch(kernel, *args, **kwargs):
    if not getattr(_state, "active", False) or not _physics(kernel):
        return _launch(kernel, *args, **kwargs)
    arena = _arena()
    if arena is not None:
        if arena.collision.handles(kernel):
            return arena.collision.run(kernel, *args, **kwargs)
        if arena.rows.handles(kernel):
            return arena.rows.run(kernel, *args, **kwargs)
    if "ccd_kernel_builder__locals__ccd_kernel" in kernel.key:
        kernel = ordered_physics_rows.ccd_kernel(kernel)
        args = _replace_dimension(args, kwargs, 1)
    elif kernel.key == "_efc_contact_pyramidal":
        kernel = ordered_physics_rows.contact_kernel(kernel)
        args = _replace_dimension(args, kwargs, 1)
    elif kernel.key == "update_constraint_efc":
        kernel = ordered_physics_rows.kernel(kernel)
        args = _replace_dimension(args, kwargs, (_dimension(args, kwargs)[0],))
    elif any(
        name in kernel.key
        for name in (
            "solve_init_jaref__locals__kernel",
            "linesearch_jv_fused__locals__kernel",
        )
    ):
        from mujoco_warp._src import solver

        nv = kernel.adj.resolve_external_reference("nv")
        factory = (
            solver.solve_init_jaref
            if "solve_init_jaref" in kernel.key
            else solver.linesearch_jv_fused
        )
        kernel = factory(nv, nv)
        args = _replace_dimension(
            args, kwargs, (*_dimension(args, kwargs)[:2], 1)
        )
    elif _shared_writes(kernel):
        kwargs["max_blocks"] = 1
        if not getattr(_state, "tiled", False):
            kwargs["block_dim"] = 1
    return _launch(kernel, *args, **kwargs)


def _ordered_tiled(kernel, *args, **kwargs):
    previous = getattr(_state, "tiled", False)
    _state.tiled = True
    try:
        if (
            getattr(_state, "active", False)
            and _physics(kernel)
            and _shared_writes(kernel)
        ):
            kwargs["max_blocks"] = 1
        return _launch_tiled(kernel, *args, **kwargs)
    finally:
        _state.tiled = previous


def check_supported(model):
    """Refuse an unvalidated backend or friction-cone specialization."""
    expected = {
        "mujoco": "3.5.0",
        "mujoco-warp": "3.5.0",
        "warp-lang": "1.11.0",
    }
    for package, required in expected.items():
        if version(package) != required:
            raise RuntimeError(
                f"Ordered physics requires {package}=={required}"
            )
    if int(model.opt.cone) != 0:
        raise RuntimeError(
            "Ordered competition physics requires pyramidal friction cones"
        )


@contextmanager
def ordered_physics(arena=None):
    """Keep other Warp users unchanged outside this thread's physics calls."""
    global _installed, _factor, _solve, _broadphase
    with _install_lock:
        if not _installed:
            import warp._src.context as context
            from mujoco_warp._src import smooth, collision_driver

            _factor = smooth._factor_i_sparse
            _solve = smooth._solve_LD_sparse
            _broadphase = collision_driver.nxn_broadphase
            smooth._factor_i_sparse = _ordered_factor
            smooth._solve_LD_sparse = _ordered_solve
            collision_driver.nxn_broadphase = _ordered_broadphase
            wp.launch = context.launch = _ordered_launch
            wp.launch_tiled = context.launch_tiled = _ordered_tiled
            _installed = True
    previous = getattr(_state, "active", False)
    previous_arena = getattr(_state, "arena", None)
    _state.active = True
    _state.arena = previous_arena if arena is None else arena
    try:
        yield
    finally:
        _state.active = previous
        _state.arena = previous_arena
