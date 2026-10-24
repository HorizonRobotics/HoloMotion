"""Deterministic sparse pyramidal Hessian gathers; scoped internal diagnostic.

One output entry owns its reduction. Contact/constraint rows retain the declared
canonical order. Existing material, controller, dynamics and loss values remain.
"""

import warp as wp


@wp.kernel
def rebuild_hessian(
    nefc_block: wp.array[int],
    block_adr: wp.array2d[int],
    block_nrow: wp.array2d[int],
    rownnz: wp.array2d[int],
    rowadr: wp.array2d[int],
    colind: wp.array3d[int],
    jac: wp.array3d[float],
    D: wp.array2d[float],
    state: wp.array2d[int],
    done: wp.array[bool],
    nv: int,
    h: wp.array3d[float],
):
    world, r, c = wp.tid()
    if r > c or r >= nv or c >= nv or done[world]:
        return
    total = h[world, r, c]
    for group in range(nefc_block[world]):
        head = block_adr[world, group]
        adr = rowadr[world, head]
        support = rownnz[world, head]
        ir = int(-1)
        ic = int(-1)
        for k in range(support):
            column = colind[world, 0, adr + k]
            if column == r:
                ir = k
            if column == c:
                ic = k
        if ir >= 0 and ic >= 0:
            low = wp.min(ir, ic)
            high = wp.max(ir, ic)
            value = float(0.0)
            for member in range(block_nrow[world, group]):
                row = head + member
                if state[world, row] == 1:
                    member_adr = rowadr[world, row]
                    value += (
                        jac[world, 0, member_adr + low]
                        * D[world, row]
                        * jac[world, 0, member_adr + high]
                    )
            if value != 0.0:
                total += value
    h[world, r, c] = total


@wp.kernel
def incremental_hessian(
    rownnz: wp.array2d[int],
    rowadr: wp.array2d[int],
    colind: wp.array3d[int],
    jac: wp.array3d[float],
    D: wp.array2d[float],
    state: wp.array2d[int],
    changed: wp.array2d[int],
    count: wp.array[int],
    nv: int,
    h: wp.array3d[float],
):
    world, r, c = wp.tid()
    if r > c or r >= nv or c >= nv:
        return
    total = h[world, r, c]
    for index in range(count[world]):
        row = changed[world, index]
        adr = rowadr[world, row]
        support = rownnz[world, row]
        ir = int(-1)
        ic = int(-1)
        for k in range(support):
            column = colind[world, 0, adr + k]
            if column == r:
                ir = k
            if column == c:
                ic = k
        if ir >= 0 and ic >= 0:
            sign = -D[world, row]
            if state[world, row] == 1:
                sign = D[world, row]
            value = (
                sign
                * jac[world, 0, adr + wp.max(ir, ic)]
                * jac[world, 0, adr + wp.min(ir, ic)]
            )
            if value != 0.0:
                total += value
    h[world, r, c] = total
