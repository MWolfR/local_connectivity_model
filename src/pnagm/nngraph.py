import numpy
import pandas

from scipy import stats, sparse
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


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

def _pick_by_distance_exponential(_idx, pts_fr, pts_to, p_pick, decay_pick):
    _p = numpy.exp(-cdist(pts_fr, pts_to)[0] / decay_pick)
    _p = p_pick * _p / _p.mean()
    return numpy.array(_idx)[numpy.random.rand(len(_p)) <= _p]

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
    A = pandas.DataFrame(pts_x[:, :3], columns=pandas.Index(["x", "y", "z"], name="coord"),
                                    index=pandas.Index(range(len(pts_x)), name="neuron"))
    B = pandas.concat([pandas.DataFrame(pts[_idx, :3], index=pandas.Index(_idx, name="i"),
                    columns=pandas.Index(["x", "y", "z"], name="coord"))
                for _idx in idx],
                axis=0, keys=range(len(idx)), names=["neuron"])
    ab_diff = threed2twod(A - B)
    angle = numpy.arctan2(ab_diff["x"], ab_diff["y"])
    return func(angle)

def point_nn_matrix(pts, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None,
                    p_pick=None, decay_pick=None,
                    angle_func=None,
                    no_diag=True, mirror=False):
    mirror = bool(mirror)
    if pts_x is None:
        pts_x = pts
        mirror = False
    shape = (len(pts_x), len(pts))

    if angle_func is None:
        angle_func = lambda a: a.apply(lambda _a: 1.0)
    if not hasattr(angle_func, "__call__"):
        if isinstance(angle_func, dict):
            angle = angle_func.get("angle", 0.0)
            offset = angle_func.get("offset", 0.0)
            angle_func = lambda a: (numpy.cos(a + angle) + 1 + offset) / (2 + offset)
        else:
            raise ValueError()
    
    kd = KDTree(pts)
    if n_neighbors is not None and dist_neighbors is None:
        if no_diag:
            _, idx = kd.query(pts_x, numpy.arange(2, 2 + n_neighbors))  # len(idx_x) x n_neighbors
        else:
            _, idx = kd.query(pts_x, numpy.arange(1, 1 + n_neighbors))  # len(idx_x) x n_neighbors
    elif dist_neighbors is not None:
        idx = list(kd.query_ball_point(pts_x, dist_neighbors))
        if no_diag:
            idx = [numpy.setdiff1d(_idx, _i) for _i, _idx in enumerate(idx)]  # This is slow. Filter after the fact?
    if p_pick is not None:
        assert p_pick <= 1.0
        if decay_pick is not None:
            idx = [
                _pick_by_distance_exponential(_idx, pts[[_i]], pts[_idx], p_pick, decay_pick)
                for _i, _idx in enumerate(idx)
            ]
        else:
            idx = [numpy.random.choice(_idx, 
                                    numpy.maximum(stats.binom(len(_idx), p_pick).rvs(), 1),
                                    replace=False)
                    if len(_idx) > 0 else numpy.array([], dtype=int)
                for _idx in idx]
    elif n_pick is not None:
        idx = [numpy.random.choice(_idx, numpy.minimum(n_pick, len(_idx)), replace=False)
               for _idx in idx]
    w = angle_based_weights(pts_x, pts, idx, angle_func)
    return to_csc_matrix(w, mirror=mirror, shape=shape)
