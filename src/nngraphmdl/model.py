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
    M = conntility.ConnectivityMatrix.from_h5(fn, "condensed")
    sz = 1000 * cfg["tgt_sz"]
    cols = ["x_nm", "y_nm", "z_nm"]
    o = pandas.Series({"x_nm": 1000 * cfg["o_x"],
                       "y_nm": 1000 * cfg["o_y"],
                       "z_nm": 1000 * cfg["o_z"]})

    center = M.vertices[cols].mean()
    N = M.index("classification_system").eq("excitatory_neuron")

    for _col in cols:
        N = N.index(_col).le(center[_col] + o[_col] + sz/2).index(_col).ge(center[_col] + o[_col] - sz/2)
    
    pts = N.vertices[cols].values / 1000
    n_nrn = len(pts)
    ss_fac = cfg["ss_fac"]
    n_sub = int(n_nrn / ss_fac)
    pts_sub = pts[numpy.random.choice(n_nrn, n_sub, replace=False), :]
    return pts, pts_sub, N

# For generating a graph connecting each neuron location to its nearest neighbors.
def threed2twod(pts, indices=[0, 2]):
    assert pts.shape[-1] == (len(indices) + 1)
    idx = numpy.setdiff1d(numpy.arange(pts.shape[-1]), indices)[0]
    tgt_shape_out = pts.shape[:-1] + (2,)
    pts = pts.reshape((-1, pts.shape[-1]))
    h = numpy.sqrt(numpy.sum(pts[:, indices] ** 2, axis=1))
    return numpy.vstack([pts[:, idx], h]).transpose().reshape(tgt_shape_out)

def angle_based_weights(pts_x, pts, idx, func):
    # 0: horizontal, pi/2: up, -pi/2: down
    A = pts_x.reshape((-1, 1, 3))
    B = numpy.dstack([pts[_idx, :] for _idx in idx])
    B = B.transpose([2, 0, 1])
    ab_diff = threed2twod(A - B)
    angle = numpy.arctan2(ab_diff[:, :, 0], ab_diff[:, :, 1])
    return func(angle)

def to_csc_matrix(idx, w, mirror=True, shape=None):
    if shape is None:
        shape = (len(idx), len(idx))
    idy = [_i * numpy.ones(len(_idx)) for _i, _idx in enumerate(idx)]
    idx = numpy.hstack(idx); idy = numpy.hstack(idy); w = numpy.hstack(w)
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

def point_nn_matrix(pts, func, pts_x=None, n_neighbors=4, n_pick=None):
    mirror = False
    if pts_x is None:
        pts_x = pts
        mirror = False
    shape = (len(pts_x), len(pts))

    kd = KDTree(pts)
    _, idx = kd.query(pts_x, numpy.arange(2, 2 + n_neighbors))  # len(idx_x) x n_neighbors
    if n_pick is not None:
        assert n_pick <= n_neighbors
        idx = [numpy.random.choice(_idx, n_pick, replace=False)
               for _idx in idx]
    w = angle_based_weights(pts_x, pts, idx, func)
    return to_csc_matrix(idx, w, mirror=mirror, shape=shape)

#######
# Entrance point 1: Generate matrix.
#######
def make_matrices(pts, pts_sub, cfg):
    prefer_down = lambda a: (numpy.cos(a + cfg["direction"]) + 1 + cfg["direction_str"]) / (2 + cfg["direction_str"])
    uniform = lambda a: numpy.ones_like(a)
    M12 = point_nn_matrix(pts_sub, uniform, pts_x=pts, n_neighbors=cfg["reverse_mapping_n"])
    M22 = adjust_p_matrix(point_nn_matrix(pts_sub, prefer_down, n_neighbors=cfg["lvl2_n"]), cfg["lvl2_fac"])
    M21 = cfg["mapping_p"] * point_nn_matrix(pts, uniform, pts_x=pts_sub, n_neighbors=cfg["mapping_n"]) / cfg["mapping_n"]
    M11 = adjust_p_matrix(point_nn_matrix(pts, uniform, n_neighbors=cfg["lvl1_n"]), cfg["lvl1_fac"])
    return M12, M22, M21, M11

## Adjusting the weights to ensure a consistent in- and out-degree.
def adjust_p_matrix(M, mul):
    exp_deg = (M > 0).sum(axis=1).mean()
    CN = (M > 0).astype(int) * (M.transpose() > 0).astype(int)
    tmp = M.tocoo()
    eff_deg = (exp_deg - CN[tmp.row, tmp.col].mean())
    deg_ratio = exp_deg / eff_deg
    return sparse.csr_matrix(mul * deg_ratio * M / M.sum(axis=1))


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
