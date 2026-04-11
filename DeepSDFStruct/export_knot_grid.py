import numpy as np
import pyvista as pv
import torch


def export_knot_grid_paramspace(spline, filename="knot_grid_paramspace.vtp"):
    """Export the tensor-product knot grid of a splinepy BSpline as a line
    network (PolyData) for ParaView.

    Parameters
    ----------
    spline : splinepy.BSpline
        The spline whose knot grid should be exported.
    filename : str
        Output file path (.vtp).
    """
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    # Unique knot values define the grid subdivisions
    u_vals = np.unique(kvs[0])
    v_vals = np.unique(kvs[1])
    w_vals = np.unique(kvs[2])

    points = []
    lines = []
    point_dict = {}

    def get_point_id(p):
        key = tuple(np.round(p, 12))
        if key not in point_dict:
            point_dict[key] = len(points)
            points.append(p)
        return point_dict[key]

    # Lines along u-direction
    for v in v_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for u in u_vals]
            lines.append([len(ids)] + ids)

    # Lines along v-direction
    for u in u_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for v in v_vals]
            lines.append([len(ids)] + ids)

    # Lines along w-direction
    for u in u_vals:
        for v in v_vals:
            ids = [get_point_id([u, v, w]) for w in w_vals]
            lines.append([len(ids)] + ids)

    points = np.array(points)
    lines = np.hstack(lines)

    grid = pv.PolyData(points)
    grid.lines = lines
    grid.save(filename)

    print(f"Knot grid exported to: {filename}")


def export_control_lattice(
    spline, filename="control_lattice_paramspace.vtp", order="F"
):
    """Export control point Greville abscissae as a point cloud with
    index metadata (id, i, j, k).

    Parameters
    ----------
    spline : splinepy.BSpline
        The spline whose control lattice should be exported.
    filename : str
        Output file path (.vtp).
    order : str
        Flattening order for control point indices ('F' or 'C').
    """
    degrees = np.array(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    def greville_1d(U, p):
        n = len(U) - p - 1
        if p == 0:
            return 0.5 * (U[:n] + U[1 : n + 1])
        return np.array(
            [np.sum(U[i + 1 : i + p + 1]) / p for i in range(n)], dtype=float
        )

    g0 = greville_1d(kvs[0], degrees[0])
    g1 = greville_1d(kvs[1], degrees[1])
    g2 = greville_1d(kvs[2], degrees[2])

    n0, n1, n2 = len(g0), len(g1), len(g2)
    N = n0 * n1 * n2
    ids = np.arange(N)

    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)
    pts = np.column_stack([g0[I], g1[J], g2[K]])

    cloud = pv.PolyData(pts)
    cloud["id"] = ids
    cloud["i"] = I
    cloud["j"] = J
    cloud["k"] = K
    cloud.save(filename)


def export_control_lattice_paramspace_locked(
    spline,
    locked_idx,
    filename="control_lattice_locked_paramspace.vtp",
    write_all_with_flag=False,
    order="F",
):
    """Export control lattice with locked control point information.

    Parameters
    ----------
    spline : splinepy.BSpline
        The spline whose control lattice should be exported.
    locked_idx : array-like
        Indices of locked (constrained) control points.
    filename : str
        Output file path (.vtp).
    write_all_with_flag : bool
        If True, exports all points with a 'locked' flag array.
        If False, exports only the locked points.
    order : str
        Flattening order for control point indices ('F' or 'C').
    """
    degrees = np.array(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    def greville_1d(U, p):
        n = len(U) - p - 1
        if n <= 0:
            raise ValueError(f"Invalid knot vector length {len(U)} for degree {p}")
        if p == 0:
            return 0.5 * (U[:n] + U[1 : n + 1])
        return np.array(
            [np.sum(U[i + 1 : i + p + 1]) / p for i in range(n)], dtype=float
        )

    g0 = greville_1d(kvs[0], degrees[0])
    g1 = greville_1d(kvs[1], degrees[1])
    g2 = greville_1d(kvs[2], degrees[2])

    n0, n1, n2 = len(g0), len(g1), len(g2)
    N = n0 * n1 * n2
    ids = np.arange(N)

    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)
    pts = np.column_stack([g0[I], g1[J], g2[K]])

    if torch.is_tensor(locked_idx):
        locked_idx = locked_idx.detach().to("cpu").numpy()
    locked_idx = np.asarray(locked_idx, dtype=int)

    if locked_idx.size == 0:
        raise ValueError("locked_idx is empty.")
    if locked_idx.min() < 0 or locked_idx.max() >= N:
        raise IndexError(f"locked_idx out of bounds. Valid range: [0, {N-1}]")

    if write_all_with_flag:
        cloud = pv.PolyData(pts)
        cloud["id"] = ids
        cloud["i"] = I
        cloud["j"] = J
        cloud["k"] = K

        locked_flag = np.zeros((N,), dtype=np.int32)
        locked_flag[locked_idx] = 1
        cloud["locked"] = locked_flag
    else:
        sel = locked_idx
        cloud = pv.PolyData(pts[sel])
        cloud["id"] = ids[sel]
        cloud["i"] = I[sel]
        cloud["j"] = J[sel]
        cloud["k"] = K[sel]

    cloud.save(filename)
    print(f"Locked control lattice exported to: {filename}")


def export_control_points(control_points, file_path):
    """Export control points as a VTP point cloud.

    Parameters
    ----------
    control_points : torch.Tensor or np.ndarray
        Shape (N, 3)
    file_path : str or Path
        Output .vtp file path
    """
    if isinstance(control_points, torch.Tensor):
        points = control_points.detach().cpu().numpy()
    else:
        points = np.asarray(control_points)

    poly = pv.PolyData(points)
    poly.save(str(file_path))
