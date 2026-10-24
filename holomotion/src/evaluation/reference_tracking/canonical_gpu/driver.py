"""Explicit instance-owned dispatch for one pinned evaluation protocol."""

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .hessian import incremental_hessian, rebuild_hessian
from .private_host import PrivateHost, WarpFacade
from .row_kernels import ActiveContactRows, BoundedRows, PyramidalOnly

VERSIONS = {"mujoco-warp": "3.16.0", "warp-lang": "1.18.0", "mujoco": "3.16.0"}


def verify_sources():
    root = Path(mjw.__file__).resolve().parent.parent
    pin = json.loads(Path(__file__).with_name("source_pin.json").read_text())
    for package, version in VERSIONS.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(
                f"canonical_gpu_v1 requires {package}=={version}"
            )
    for name, expected in pin.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"canonical_gpu_v1 source mismatch: {name}")
    return hashlib.sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()


def verify_model(cpu_model, expected_digest):
    actual = hashlib.sha256(cpu_model.__getstate__()).hexdigest()
    if actual != expected_digest:
        raise ValueError(f"canonical_gpu_v1 model/options mismatch: {actual}")
    return actual


class CanonicalDriver:
    def __init__(self, cpu_model, *, evaluation):
        if not evaluation:
            raise ValueError("canonical_gpu_v1 is evaluation-only")
        if cpu_model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
            raise ValueError("canonical_gpu_v1 requires a pyramidal model")
        self.source_digest = verify_sources()
        self.protocol_pin = json.loads(
            Path(__file__).with_name("protocol_pin.json").read_text()
        )
        self.model_digest = verify_model(
            cpu_model, self.protocol_pin["model_sha256"]
        )
        # This exact model has no flex geometry or negative geometry IDs.
        if cpu_model.ngeom != 72 or cpu_model.nflex != 0:
            raise ValueError(
                "Unsupported canonical contact key geometry range"
            )
        self.namespace = f"_canonical_instance_{id(self):x}"
        self.clones = {}
        self.stats = {}
        self.host = PrivateHost(WarpFacade(wp, self.launch, self.launch_tiled))
        from mujoco_warp._src import constraint, forward

        self.constraint = self.host.module(constraint)
        native_constraint = self.constraint.make_constraint

        def sorted_constraint(model, data):
            self.sort_contacts(data)
            return native_constraint(model, data)

        self.constraint.make_constraint = sorted_constraint
        self.forward = self.host.module(forward)
        self.step = self.forward.step
        spec = json.loads(
            Path(__file__).with_name("dispatch_pin.json").read_text()
        )
        self.modified = set(spec["modified"])
        self.world_rows = set(spec["world_rows"])
        self.known = set(spec["known"])
        self.tiled = set(spec["tiled"])
        self.tiled_stats = {}

    def sort_contacts(self, data):
        contact = data.contact
        arrays = {
            f.name: wp.to_torch(getattr(contact, f.name))
            for f in dataclasses.fields(contact)
        }
        capacity = arrays["worldid"].shape[0]
        order = torch.arange(capacity, device=arrays["worldid"].device)
        valid = order < wp.to_torch(data.nacon)[0]
        # The source/model pins guarantee valid rows have worldid in
        # [0, nworld) and geom IDs in [0, 72). Mixed radix preserves the
        # exact lexicographic integer order; invalid rows share one sentinel.
        invalid_key = data.nworld * 72 * 72
        if not 0 < invalid_key <= torch.iinfo(torch.int32).max:
            raise RuntimeError("Unsupported canonical contact world range")
        integer_key = torch.where(
            valid,
            (arrays["worldid"] * 72 + arrays["geom"][:, 0]) * 72
            + arrays["geom"][:, 1],
            invalid_key,
        )
        keys = [
            arrays["geomcollisionid"],
            *[arrays["pos"][:, k] for k in (2, 1, 0)],
            arrays["dist"],
        ]
        for key in keys:
            safe = torch.where(valid, key, torch.zeros_like(key))
            order = order[torch.argsort(safe[order], stable=True)]
        order = order[torch.argsort(integer_key[order], stable=True)]
        for name, array in arrays.items():
            if array.shape[0] == capacity:
                array.copy_(array[order])
            elif array.shape[0] != 0:
                raise RuntimeError(
                    f"Unexpected canonical contact field {name}"
                )

    def clone(self, kernel, dimensions, pyramidal=False):
        key = (id(kernel), dimensions, pyramidal)
        if key not in self.clones:
            token = hashlib.sha256(
                repr((kernel.key, dimensions, pyramidal)).encode()
            ).hexdigest()[:12]
            module = wp.get_module(
                kernel.module.name + self.namespace + "_" + token
            )
            wp.set_module_options(dict(kernel.module.options), module=module)
            transforms = ([PyramidalOnly()] if pyramidal else []) + [
                BoundedRows(dimensions)
            ]
            self.clones[key] = wp.Kernel(
                kernel.func,
                key=kernel.key + "_canonical_" + token,
                module=module,
                options={"enable_backward": False, "grid_stride": False},
                code_transformers=transforms,
            )
        return self.clones[key]

    def launch(self, kernel, *args, **kwargs):
        key = kernel.key
        if key not in self.known:
            raise RuntimeError(f"Unqualified canonical kernel: {key}")
        self.stats[key] = self.stats.get(key, 0) + 1
        hessian = key in (
            "_JTDACJ_sparse__locals__kernel",
            "_update_gradient_h_incremental_sparse__locals__kernel",
        )
        if hessian:
            rebuild = key == "_JTDACJ_sparse__locals__kernel"
            if kernel.adj.resolve_external_reference(
                "COMPACT"
            ) is not False or (
                rebuild
                and kernel.adj.resolve_external_reference("ELLIPTIC")
                is not False
            ):
                raise RuntimeError("Unsupported canonical Hessian layout")
            v = dict(
                zip(
                    inspect.signature(kernel.func).parameters,
                    [*kwargs.get("inputs", []), *kwargs.get("outputs", [])],
                )
            )
            nv = v["dof_cdof_in"].shape[1]
            h = v["h_out" if rebuild else "ctx_h_out"]
            names = (
                [
                    "efc_jtdaj_nblock_in",
                    "efc_jtdaj_adr_in",
                    "efc_jtdaj_nrow_in",
                    "efc_J_rownnz_in",
                    "efc_J_rowadr_in",
                    "efc_J_colind_in",
                    "efc_J_in",
                    "efc_D_in",
                    "efc_state_in",
                    "ctx_done_in",
                ]
                if rebuild
                else [
                    "efc_J_rownnz_in",
                    "efc_J_rowadr_in",
                    "efc_J_colind_in",
                    "efc_J_in",
                    "efc_D_in",
                    "efc_state_in",
                    "quad_changed_ids_in",
                    "quad_changed_count_in",
                ]
            )
            return wp.launch(
                rebuild_hessian if rebuild else incremental_hessian,
                dim=(h.shape[0], nv, nv),
                inputs=[v[n] for n in names] + [nv],
                outputs=[h],
            )
        dimension = kwargs.get("dim", args[0] if args else None)
        if key == "_efc_contact_init__locals__kernel":
            if not isinstance(dimension, int) or dimension < 1:
                raise RuntimeError("Unexpected canonical contact capacity")
            cache_key = (id(kernel), "active_contacts", dimension)
            if cache_key not in self.clones:
                module = wp.get_module(
                    kernel.module.name
                    + self.namespace
                    + "_active_contacts_"
                    + str(dimension)
                )
                wp.set_module_options(
                    dict(kernel.module.options), module=module
                )
                self.clones[cache_key] = wp.Kernel(
                    kernel.func,
                    key=kernel.key + "_active_contacts_" + str(dimension),
                    module=module,
                    options={"enable_backward": False, "grid_stride": False},
                    code_transformers=[ActiveContactRows(dimension)],
                )
            kwargs.pop("max_blocks", None)
            kwargs["block_dim"] = 1
            if "dim" in kwargs:
                kwargs["dim"] = 1
            else:
                args = (1, *args[1:])
            return wp.launch(self.clones[cache_key], *args, **kwargs)
        # Upstream uses a list for this world/node launch. Preserve ascending
        # node order within each world, while independent worlds run in parallel.
        if key == "_cfrc_backward":
            if not isinstance(dimension, (list, tuple)) or len(dimension) != 2:
                raise RuntimeError(
                    "Unexpected canonical cfrc world/node shape"
                )
            dimension = tuple(dimension)
        pyramidal = key == "_update_constraint_efc__locals__kernel"
        selected = key in self.modified
        if key == "_update_gradient_zero_grad_dot__locals__kernel":
            selected = bool(
                kernel.adj.resolve_external_reference("STABLE_FAST")
            )
        if pyramidal or (selected and key in self.world_rows):
            if not isinstance(dimension, tuple) or len(dimension) < 2:
                raise RuntimeError(f"Unexpected canonical launch shape: {key}")
            kernel = self.clone(
                kernel, tuple(int(d) for d in dimension[1:]), pyramidal
            )
            kwargs.pop("max_blocks", None)
            if "dim" in kwargs:
                kwargs["dim"] = (dimension[0],)
            else:
                args = ((dimension[0],), *args[1:])
        elif selected:
            cache_key = (id(kernel), "serial")
            if cache_key not in self.clones:
                self.clones[cache_key] = wp.kernel(
                    kernel.func,
                    module="unique",
                    module_options=dict(
                        kernel.module.options,
                        default_grid_stride=True,
                        deterministic=wp.DeterministicMode.NOT_GUARANTEED,
                    ),
                    grid_stride=True,
                    enable_backward=False,
                    launch_bounds=kernel.options.get("launch_bounds"),
                )
            kernel = self.clones[cache_key]
            kwargs.update(max_blocks=1, block_dim=1)
        return wp.launch(kernel, *args, **kwargs)

    def metadata(self):
        root = Path(__file__).parent
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix in (".py", ".json")
        }
        return {
            "mode": "canonical_gpu_v1",
            "new_evaluation_protocol": True,
            "fresh_reference_baseline_required": True,
            "model_sha256": self.model_digest,
            "installed_source_pin_sha256": self.source_digest,
            "implementation_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
            "implementation_files": files,
            "versions": VERSIONS,
            "cross_host_fingerprint_qualified": False,
            "actual_tiled_dispatch_counts": dict(self.tiled_stats),
        }

    def launch_tiled(self, kernel, *args, **kwargs):
        if kernel.key not in self.tiled:
            raise RuntimeError(
                f"Unqualified canonical tiled kernel: {kernel.key}"
            )
        self.tiled_stats[kernel.key] = self.tiled_stats.get(kernel.key, 0) + 1
        if kernel.key in self.modified:
            raise RuntimeError(
                f"Unqualified canonical tiled reduction: {kernel.key}"
            )
        return wp.launch_tiled(kernel, *args, **kwargs)
