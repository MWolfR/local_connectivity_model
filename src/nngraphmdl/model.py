import numpy
import pandas

from scipy import sparse
from scipy import stats
from scipy.spatial import KDTree

# For the random or non-random generation of neuron locations in space
def make_points(cfg):
    n_nrn = cfg["n_nrn"]
    ss_fac = cfg["ss_fac"]
    tgt_sz = cfg["tgt_sz"] 
    n_sub = int(n_nrn / ss_fac)
    pts = numpy.random.rand(n_nrn, 3) * tgt_sz - tgt_sz/2
    pts_sub = numpy.random.rand(n_sub, 3) * tgt_sz - tgt_sz/2
    return pts, pts_sub

def points_from_microns(cfg):
    import conntility
    fn = cfg["fn"]
    N = conntility.ConnectivityMatrix.from_h5(fn, "condensed")
    sz = 1000 * cfg["tgt_sz"]
    cols = ["x_nm", "y_nm", "z_nm"]

    center = N.vertices[cols].mean()
    for k, v in cfg.get("filters", {"classification_system": "excitatory_neuron"}).items():
        if isinstance(v, list):
            N = N.index(k).isin(v)
        else:
            N = N.index(k).eq(v)

    for _col in cols:
        _col_o = "o_" + _col[0]
        if _col_o in cfg:
            _o = 1000 * cfg[_col_o]
            N = N.index(_col).le(center[_col] + _o + sz/2).index(_col).ge(center[_col] + _o - sz/2)
            print(len(N))
    
    pts = N.vertices[cols].values / 1000
    n_nrn = len(pts)
    ss_fac = cfg["ss_fac"]
    n_sub = int(n_nrn / ss_fac)
    pts_sub = pts[numpy.random.choice(n_nrn, n_sub, replace=False), :]
    return pts, pts_sub, N

# For generating a graph connecting each neuron location to its nearest neighbors.
def threed2twod(pts, indices=["x", "z"]):
    col_keep = [_col for _col in pts.columns if _col not in indices]
    assert len(col_keep) == 1
    col_keep = col_keep[0]
    h = numpy.sqrt(numpy.sum(pts[indices].values ** 2, axis=1))
    return pandas.DataFrame({
        "x": pts[col_keep].values,
        "y": h
    }, index=pts.index)

def angle_based_weights(pts_x, pts, idx, func):
    # 0: horizontal, pi/2: up, -pi/2: down
    A = pandas.DataFrame(pts_x, columns=pandas.Index(["x", "y", "z"], name="coord"),
                                    index=pandas.Index(range(len(pts_x)), name="neuron"))
    B = pandas.concat([pandas.DataFrame(pts[_idx, :], index=pandas.Index(_idx, name="i"),
                    columns=pandas.Index(["x", "y", "z"], name="coord"))
                for _idx in idx],
                axis=0, keys=range(len(idx)), names=["neuron"])
    ab_diff = threed2twod(A - B)
    angle = numpy.arctan2(ab_diff["x"], ab_diff["y"])
    return func(angle)

def to_csc_matrix(w, mirror=True, shape=None):
    if shape is None:
        raise ValueError("Must provide shape")
    indices = w.index.to_frame().reset_index(drop=True)
    idy = indices["neuron"].values
    idx = indices["i"].values
    w = w.values
    if mirror:
        _idx = numpy.hstack([idx, idy])
        _idy = numpy.hstack([idy, idx])
        _w = numpy.hstack([w, w])
        M = sparse.coo_matrix((_w, (_idy, _idx)),
                              shape=shape).tocsc()
    else:
        M = sparse.coo_matrix((w, (idy, idx)),
                              shape=shape).tocsc()
    return M

def point_nn_matrix(pts, func, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None, no_diag=True):
    mirror = False
    if pts_x is None:
        pts_x = pts
        mirror = False
    shape = (len(pts_x), len(pts))

    kd = KDTree(pts)
    if n_neighbors is not None:
        if no_diag:
            _, idx = kd.query(pts_x, numpy.arange(2, 2 + n_neighbors))  # len(idx_x) x n_neighbors
        else:
            _, idx = kd.query(pts_x, numpy.arange(1, 1 + n_neighbors))  # len(idx_x) x n_neighbors
    elif dist_neighbors is not None:
        idx = list(kd.query_ball_point(pts_x, dist_neighbors))
        if no_diag:
            idx = [numpy.setdiff1d(_idx, _i) for _i, _idx in enumerate(idx)]  # This is slow. Filter after the fact?
    if n_pick is not None:
        idx = [numpy.random.choice(_idx, numpy.minimum(n_pick, len(_idx)), replace=False)
               for _idx in idx]
    w = angle_based_weights(pts_x, pts, idx, func)
    return to_csc_matrix(w, mirror=mirror, shape=shape)

#######
# Entrance point 1: Generate matrix.
#######
def make_matrices(pts, pts_sub, cfg):
    prefer_down = lambda a: (numpy.cos(a + cfg["direction"]) + 1 + cfg["direction_str"]) / (2 + cfg["direction_str"])
    uniform = lambda a: a.apply(lambda _a: 1.0)
    M12 = point_nn_matrix(pts_sub, uniform, pts_x=pts, n_neighbors=cfg["reverse_mapping_n"], no_diag=False)
    M22 = adjust_p_matrix(point_nn_matrix(pts_sub, prefer_down,
                                          n_neighbors=cfg.get("lvl2_n"), dist_neighbors=cfg.get("lvl2_dist")),
                          cfg["lvl2_fac"])
    M21 = cfg["mapping_p"] * point_nn_matrix(pts, uniform, pts_x=pts_sub, n_neighbors=cfg["mapping_n"], no_diag=False) / cfg["mapping_n"]
    M11 = adjust_p_matrix(point_nn_matrix(pts, uniform,
                                          n_neighbors=cfg.get("lvl1_n"), dist_neighbors=cfg.get("lvl1_dist")),
                          cfg["lvl1_fac"])
    return M12, M22, M21, M11

## Adjusting the weights to ensure a consistent in- and out-degree.
def adjust_p_matrix(M, mul):
    exp_deg = (M > 0).sum(axis=1).mean()
    CN = (M > 0).astype(int) * (M.transpose() > 0).astype(int)
    tmp = M.tocoo()
    eff_deg = (exp_deg - CN[tmp.row, tmp.col].mean())
    deg_ratio = exp_deg / eff_deg
    return sparse.csr_matrix(mul * deg_ratio * M / (M.sum(axis=1) + 1E-12))


# For wiring up the model, based on the nearest neighbor graph.
def evaluate_probs(p_mat, adjust=None):
    p_mat = p_mat.tocoo()
    thresh = p_mat.data
    if adjust is not None:
        thresh = thresh * adjust[p_mat.row]
    _v = numpy.random.rand(p_mat.nnz) < thresh
    return sparse.coo_matrix((numpy.ones(_v.sum()), (p_mat.row[_v], p_mat.col[_v])), shape=p_mat.shape)

#######
# Entrance point 2: Wire up the model.
#######
def wire(M12, M22, M21, M11, pts, pts_sub, cfg):
    exclusion = sparse.coo_matrix(([], ([], [])), shape=(len(pts), len(pts_sub)))

    row = []; col = []
    
    foo = M12.tocoo()
    exclusion = exclusion + foo
    counter = []
    lim = cfg["lvl2_lim"]
    
    for _ in range(cfg["lvl2_steps"]):
        currsum = numpy.array(foo.sum(axis=1))[:, 0]
        adjust = numpy.minimum(numpy.exp(-(currsum/lim - 1)), 1.0)
        row.extend(foo.row)
        col.extend(foo.col)
        counter.append(pandas.Series(foo.row).value_counts())
    
        foo = foo * M22
        _foo = (foo - exclusion).tocoo()
        exclusion = exclusion + foo
        foo = evaluate_probs(_foo, adjust=adjust)
    
    interm = sparse.coo_matrix((numpy.ones(len(row)), (row, col)), shape=_foo.shape).tocsr()
    exclusion = sparse.coo_matrix(([], ([], [])), shape=(len(pts), len(pts)))
    row = []; col = []
    
    foo = evaluate_probs(interm * M21)
    exclusion = exclusion + foo
    counter2 = []
    lim = cfg["lvl1_lim"]
    
    for _ in range(cfg["lvl1_steps"]):
        currsum = numpy.array(foo.sum(axis=1))[:, 0]
        adjust = numpy.minimum(numpy.exp(-(currsum/lim - 1)), 1.0)
        row.extend(foo.row)
        col.extend(foo.col)
        counter2.append(pandas.Series(foo.row).value_counts())
    
        foo = foo * M11
        _foo = (foo - exclusion).tocoo()
        exclusion = exclusion + foo
        foo = evaluate_probs(_foo, adjust=adjust)

    row = numpy.array(row); col = numpy.array(col)
    _v = row != col
    row = row[_v]; col=col[_v]
    final = sparse.coo_matrix((numpy.ones(len(row)), (row, col)), shape=_foo.shape).tocsr() > 0
    if cfg["p_final"] < 1:
        final = evaluate_probs(final * cfg["p_final"]).astype(bool)
    return final, pandas.concat(counter, axis=1).fillna(0).sort_index(), pandas.concat(counter2, axis=1).fillna(0).sort_index()
