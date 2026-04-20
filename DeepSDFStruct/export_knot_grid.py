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

    for v in v_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for u in u_vals]
            lines.append([len(ids)] + ids)

    for u in u_vals:
        for w in w_vals:
            ids = [get_point_id([u, v, w]) for v in v_vals]
            lines.append([len(ids)] + ids)

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


def _greville_1d(U, p):
    n = len(U) - p - 1
    if n <= 0:
        raise ValueError(f"Invalid knot vector length {len(U)} for degree {p}")
    if p == 0:
        return 0.5 * (U[:n] + U[1 : n + 1])
    return np.array(
        [np.sum(U[i + 1 : i + p + 1]) / p for i in range(n)], dtype=float
    )


def export_control_lattice_paramspace(
    spline,
    filename="control_lattice_paramspace.vtp",
    locked_idx=None,
    order="F",
):
    """Export control-lattice Greville abscissae (parametric space) as a
    .vtp point cloud with id/i/j/k metadata. If locked_idx is given, adds
    a `locked` int32 point array (0 = free, 1 = locked) so ParaView can
    color free vs. locked control points.

    Parameters
    ----------
    spline : splinepy.BSpline
    filename : str
        Output file path (.vtp).
    locked_idx : array-like or torch.Tensor, optional
        Indices of locked control points. If None or empty, the `locked`
        array is omitted.
    order : str
        Flattening order for control point indices ('F' or 'C').
    """
    degrees = np.array(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    g0 = _greville_1d(kvs[0], degrees[0])
    g1 = _greville_1d(kvs[1], degrees[1])
    g2 = _greville_1d(kvs[2], degrees[2])

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

    if locked_idx is not None:
        if torch.is_tensor(locked_idx):
            locked_idx = locked_idx.detach().to("cpu").numpy()
        locked_idx = np.asarray(locked_idx, dtype=int)
        if locked_idx.size > 0:
            if locked_idx.min() < 0 or locked_idx.max() >= N:
                raise IndexError(
                    f"locked_idx out of bounds. Valid range: [0, {N-1}]"
                )
            locked_flag = np.zeros((N,), dtype=np.int32)
            locked_flag[locked_idx] = 1
            cloud["locked"] = locked_flag

    cloud.save(filename)
    print(f"Control lattice exported to: {filename}")


def export_control_lattice_physical(
    control_points,
    n_per_dim,
    filename,
    order="F",
):
    """Export a deformed control lattice in physical space: points plus
    lines connecting (i,j,k) neighbors along each grid axis.

    Parameters
    ----------
    control_points : torch.Tensor or np.ndarray
        Shape (N, 3) where N == prod(n_per_dim).
    n_per_dim : sequence of 3 ints
        (n0, n1, n2) grid shape of the control lattice.
    filename : str or Path
        Output .vtp file path.
    order : str
        Flattening order for control point indices ('F' or 'C').
    """
    if isinstance(control_points, torch.Tensor):
        pts = control_points.detach().cpu().numpy()
    else:
        pts = np.asarray(control_points)

    n0, n1, n2 = (int(n) for n in n_per_dim)
    N = n0 * n1 * n2
    if pts.shape[0] != N:
        raise ValueError(
            f"control_points has {pts.shape[0]} rows but n_per_dim={n_per_dim} implies {N}"
        )

    ids = np.arange(N)
    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)

    def flat(i, j, k):
        return np.ravel_multi_index((i, j, k), (n0, n1, n2), order=order)

    edges = []
    # +i edges
    if n0 > 1:
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(n0 - 1), np.arange(n1), np.arange(n2), indexing="ij",
        )
        a = flat(i_idx.ravel(), j_idx.ravel(), k_idx.ravel())
        b = flat(i_idx.ravel() + 1, j_idx.ravel(), k_idx.ravel())
        edges.append(np.column_stack([a, b]))
    # +j edges
    if n1 > 1:
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(n0), np.arange(n1 - 1), np.arange(n2), indexing="ij",
        )
        a = flat(i_idx.ravel(), j_idx.ravel(), k_idx.ravel())
        b = flat(i_idx.ravel(), j_idx.ravel() + 1, k_idx.ravel())
        edges.append(np.column_stack([a, b]))
    # +k edges
    if n2 > 1:
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(n0), np.arange(n1), np.arange(n2 - 1), indexing="ij",
        )
        a = flat(i_idx.ravel(), j_idx.ravel(), k_idx.ravel())
        b = flat(i_idx.ravel(), j_idx.ravel(), k_idx.ravel() + 1)
        edges.append(np.column_stack([a, b]))

    if edges:
        edge_array = np.vstack(edges)
        lines = np.column_stack(
            [np.full(edge_array.shape[0], 2, dtype=np.int64), edge_array]
        ).ravel()
    else:
        lines = np.array([], dtype=np.int64)

    poly = pv.PolyData(pts)
    if lines.size > 0:
        poly.lines = lines
    poly["id"] = ids
    poly["i"] = I
    poly["j"] = J
    poly["k"] = K
    poly.save(str(filename))
    print(f"Control lattice (physical) exported to: {filename}")


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
