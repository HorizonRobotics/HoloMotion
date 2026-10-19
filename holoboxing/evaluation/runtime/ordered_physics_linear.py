"""Sparse factor/solve with fixed per-output order in one CUDA block.

Different output cells run in parallel. Contributions to each cell retain the
upstream group and update order, including float32 product/subtract rounding.
Every thread participates in the barriers; launch exactly one 128-thread block.
"""

import numpy as np
import warp as wp


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def barrier():
    pass


@wp.kernel
def factor_groups(
    rowadr: wp.array(dtype=int),
    rownnz: wp.array(dtype=int),
    updates: wp.array(dtype=wp.vec3i),
    update_offsets: wp.array(dtype=int),
    groups: wp.array(dtype=int),
    dest: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    source: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    tmp: wp.array(dtype=float),
    L: wp.array3d(dtype=float),
    D: wp.array2d(dtype=float),
):
    lane = wp.tid()
    for group in range(groups.shape[0] - 2, -1, -1):
        for node in range(
            update_offsets[group] + lane, update_offsets[group + 1], 128
        ):
            u = updates[node]
            tmp[node] = (
                L[0, 0, u[2]] / L[0, 0, rowadr[u[1]] + rownnz[u[1]] - 1]
            )
        barrier()
        for cell in range(groups[group] + lane, groups[group + 1], 128):
            address = dest[cell]
            for j in range(offsets[cell], offsets[cell + 1]):
                # A single lane owns this cell. The atomic retains the original
                # separate product/subtraction rounding, without an FMA rewrite.
                wp.atomic_sub(
                    L[0, 0], address, L[0, 0, source[j]] * tmp[nodes[j]]
                )
        barrier()
        for node in range(
            update_offsets[group] + lane, update_offsets[group + 1], 128
        ):
            L[0, 0, updates[node][2]] = tmp[node]
        barrier()
    for dof in range(lane, D.shape[1], 128):
        D[0, dof] = 1.0 / L[0, 0, rowadr[dof] + rownnz[dof] - 1]


@wp.kernel
def solve_groups(
    up_groups: wp.array(dtype=int),
    up_dest: wp.array(dtype=int),
    up_offsets: wp.array(dtype=int),
    up_source: wp.array(dtype=int),
    up_addresses: wp.array(dtype=int),
    down_groups: wp.array(dtype=int),
    down_dest: wp.array(dtype=int),
    down_offsets: wp.array(dtype=int),
    down_source: wp.array(dtype=int),
    down_addresses: wp.array(dtype=int),
    L: wp.array3d(dtype=float),
    D: wp.array2d(dtype=float),
    x: wp.array2d(dtype=float),
):
    lane = wp.tid()
    for group in range(up_groups.shape[0] - 2, -1, -1):
        for cell in range(up_groups[group] + lane, up_groups[group + 1], 128):
            address = up_dest[cell]
            for j in range(up_offsets[cell], up_offsets[cell + 1]):
                wp.atomic_sub(
                    x[0],
                    address,
                    L[0, 0, up_addresses[j]] * x[0, up_source[j]],
                )
        barrier()
    for dof in range(lane, D.shape[1], 128):
        x[0, dof] *= D[0, dof]
    barrier()
    for group in range(down_groups.shape[0] - 1):
        for cell in range(
            down_groups[group] + lane, down_groups[group + 1], 128
        ):
            address = down_dest[cell]
            for j in range(down_offsets[cell], down_offsets[cell + 1]):
                wp.atomic_sub(
                    x[0],
                    address,
                    L[0, 0, down_addresses[j]] * x[0, down_source[j]],
                )
        barrier()


def contribution_plan(contributions):
    """Stable CSR lists; never sort or regroup a cell's contributions."""
    dest, offsets, source, nodes = list(contributions), [0], [], []
    for target in dest:
        for src, node in contributions[target]:
            source.append(src)
            nodes.append(node)
        offsets.append(len(source))
    return dest, offsets, source, nodes


def build_plans(rowadr, rownnz, update_groups):
    """Prove that each group's reads and writes can execute independently."""
    plans = []
    for updates in update_groups:
        targets, sources = set(updates[:, 0]), set(updates[:, 1])
        if targets & sources:
            raise ValueError("Sparse update group has a read/write dependency")
        if len(set(updates[:, 2])) != len(updates):
            raise ValueError("Sparse multiplier addresses must be unique")
        matrix, up, down = {}, {}, {}
        for node, (i, k, ki) in enumerate(updates):
            if rownnz[i] > rownnz[k] or not (
                rowadr[k] <= ki < rowadr[k] + rownnz[k] - 1
            ):
                raise ValueError("Unsupported sparse row layout")
            for j in range(rownnz[i]):
                matrix.setdefault(int(rowadr[i] + j), []).append(
                    (int(rowadr[k] + j), node)
                )
            up.setdefault(int(i), []).append((int(k), int(ki)))
            down.setdefault(int(k), []).append((int(i), int(ki)))
        plans.append(tuple(contribution_plan(x) for x in (matrix, up, down)))
    return plans


def join_plans(plans, device, node_offsets=None):
    groups, dest, offsets, source, nodes = [0], [], [0], [], []
    for index, (dd, oo, ss, nn) in enumerate(plans):
        dest.extend(dd)
        groups.append(len(dest))
        offsets.extend(x + len(source) for x in oo[1:])
        source.extend(ss)
        base = node_offsets[index] if node_offsets is not None else 0
        nodes.extend(x + base for x in nn)
    return tuple(
        wp.array(np.asarray(x, dtype=np.int32), dtype=int, device=device)
        for x in (groups, dest, offsets, source, nodes)
    )


class SparsePlan:
    def __init__(self, model, launch):
        self.model, self.launch = model, launch
        device = model.M_rowadr.device
        updates = [array.numpy() for array in model.qLD_updates]
        if not updates:
            raise ValueError("Ordered arena requires nonempty sparse updates")
        plans = build_plans(
            model.M_rowadr.numpy(), model.M_rownnz.numpy(), updates
        )
        offsets = np.cumsum([0, *[len(x) for x in updates]], dtype=np.int32)
        self.updates = wp.array(
            np.concatenate(updates), dtype=wp.vec3i, device=device
        )
        self.indices = wp.array(offsets, dtype=int, device=device)
        self.matrix = join_plans([p[0] for p in plans], device, offsets)
        self.up = join_plans([p[1] for p in plans], device)
        self.down = join_plans([p[2] for p in plans], device)
        self.tmp = wp.empty(self.updates.shape[0], dtype=float, device=device)

    def factor(self, model, data, matrix, factor, diagonal):
        from mujoco_warp._src import smooth

        if model is not self.model or data.nworld != 1:
            raise ValueError("Ordered factor requires its single arena")
        self.launch(
            smooth._copy_CSR,
            dim=(1, model.nC),
            inputs=[model.mapM2M, matrix],
            outputs=[factor],
        )
        self.launch(
            factor_groups,
            dim=128,
            block_dim=128,
            max_blocks=1,
            inputs=[
                model.M_rowadr,
                model.M_rownnz,
                self.updates,
                self.indices,
                *self.matrix,
                self.tmp,
                factor,
                diagonal,
            ],
        )

    def solve(self, model, data, factor, diagonal, solution, right_hand_side):
        if model is not self.model or data.nworld != 1:
            raise ValueError("Ordered solve requires its single arena")
        wp.copy(solution, right_hand_side)
        self.launch(
            solve_groups,
            dim=128,
            block_dim=128,
            max_blocks=1,
            inputs=[*self.up, *self.down, factor, diagonal, solution],
        )
