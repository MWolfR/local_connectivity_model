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

def distance_based_weights(pts_x, pts, idx, func):
    A = pandas.DataFrame(pts_x[:, :3], columns=pandas.Index(["x", "y", "z"], name="coord"),
                                    index=pandas.Index(range(len(pts_x)), name="neuron"))
    B = pandas.concat([pandas.DataFrame(pts[_idx, :3], index=pandas.Index(_idx, name="i"),
                    columns=pandas.Index(["x", "y", "z"], name="coord"))
                for _idx in idx],
                axis=0, keys=range(len(idx)), names=["neuron"])
    ab_diff = A - B
    ab_diff = numpy.sqrt((ab_diff ** 2).sum(axis=1))
    return func(ab_diff)

def non_isotropic_dist(pts_x, pts, idx, directionality_fac=0, directionality_axis=None, distance_func=None):
    if distance_func is None:
        distance_func = lambda _x: 1.0
    A = pandas.DataFrame(pts_x[:, :3], columns=pandas.Index(["x", "y", "z"], name="coord"),
                                    index=pandas.Index(range(len(pts_x)), name="neuron"))
    B = pandas.concat([pandas.DataFrame(pts[_idx, :3], index=pandas.Index(_idx, name="i"),
                    columns=pandas.Index(["x", "y", "z"], name="coord"))
                for _idx in idx],
                axis=0, keys=range(len(idx)), names=["neuron"])
    ab_diff = A - B
    
    pairw_D = pandas.Series(numpy.linalg.norm(ab_diff, axis=1), index=ab_diff.index)

    if directionality_axis is not None:
        directionality_axis = numpy.array(directionality_axis).reshape((-1, 1))
        align = pandas.Series(numpy.dot(ab_diff, directionality_axis)[:, 0],
                              index=ab_diff.index) / pairw_D
        return (align * directionality_fac + 1) * pairw_D.apply(distance_func)

    return pairw_D.apply(distance_func)


def generate_custom_weights_by_node_class(reference, property_to_use, axis):
    c = reference.vertices[property_to_use].value_counts().sort_index()
    nrml = c.values.reshape((-1, 1)) * c.values.reshape((1, -1))

    MM = reference.condense(property_to_use).array / nrml

    w_per_class = MM.mean(axis=axis) / numpy.sqrt(MM.mean())
    w_per_class = pandas.Series(w_per_class, name="weight", index=c.index)
    w_per_node = w_per_class[reference.vertices[property_to_use]]
    return w_per_node


def custom_weight_evaluation(w_out, w_in, idx):
    w_out = numpy.array(w_out)
    w_in = numpy.array(w_in)

    W = [pandas.Series(_w_o * w_in[_idx], name="weight",
                       index=pandas.Index(_idx, name="i"))
         for _w_o, _idx in zip(w_out, idx)]
    W = pandas.concat(W, axis=0, keys=range(len(idx)), names=["neuron"])
    return W


def point_nn_matrix(pts, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None,
                    p_pick=None, decay_pick=None,
                    angle_func=None, scale_axes=None,
                    no_diag=True, mirror=False):
    mirror = bool(mirror)
    if pts_x is None:
        pts_x = pts
        mirror = False
    if scale_axes is not None:
        pts = pts / numpy.array(scale_axes).reshape((1, -1))
        pts_x = pts_x / numpy.array(scale_axes).reshape((1, -1))
    shape = (len(pts_x), len(pts))

    if angle_func is None:
        angle_func = lambda a: a.apply(lambda _a: 1.0)
    if not hasattr(angle_func, "__call__"):
        if isinstance(angle_func, dict):
            angle_param = angle_func.get("angle", 0.0)
            offset_param = angle_func.get("offset", 0.0)
            angle_func = lambda a: (numpy.cos(a + angle_param) + 1 + offset_param) / (2 + offset_param)
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

def cand_point_nn_matrix(pts, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None,
                    p_pick=None, scale_axes=None,
                    angle_func=None, distance_func=None,
                    no_diag=True, mirror=False):
    mirror = bool(mirror)
    if pts_x is None:
        pts_x = pts
        mirror = False
    if scale_axes is not None:
        pts = pts / numpy.array(scale_axes).reshape((1, -1))
        pts_x = pts_x / numpy.array(scale_axes).reshape((1, -1))
    shape = (len(pts_x), len(pts))

    if distance_func is None:
        distance_func = lambda a: a.apply(lambda _a: 1.0)
    if angle_func is None:
        angle_func = lambda a: a.apply(lambda _a: 1.0)
    if not hasattr(angle_func, "__call__"):
        if isinstance(angle_func, dict):
            angle_param = angle_func.get("angle", 0.0)
            offset_param = angle_func.get("offset", 0.0)
            angle_func = lambda a: (numpy.cos(a + angle_param) + 1 + offset_param) / (2 + offset_param)
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
    w = angle_based_weights(pts_x, pts, idx, angle_func) *\
        distance_based_weights(pts_x, pts, idx, distance_func)
    
    if p_pick is not None:
        assert p_pick <= 1.0
        wmean = w.mean()
        picker_func = lambda _x: _x.loc[numpy.random.rand(len(_x)) < (p_pick * _x / wmean)]
    elif n_pick is not None:
        picker_func = lambda _x: _x.iloc[numpy.random.choice(len(_x), numpy.minimum(n_pick, len(_x)),
                                                             replace=False,
                                                             p=_x / _x.sum())]
    w = w.groupby("neuron").apply(picker_func).droplevel(0)
    return to_csc_matrix(w, mirror=mirror, shape=shape)


def cand2_point_nn_matrix(pts, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None,
                    p_pick=None, scale_axes=None,
                    directionality_fac=0.0, directionality_axis=None,
                    distance_func=None,
                    custom_w_out=None, custom_w_in=None,
                    no_diag=True, mirror=False):
    mirror = bool(mirror)
    if pts_x is None:
        pts_x = pts
        mirror = False
    if scale_axes is not None:
        pts = pts / numpy.array(scale_axes).reshape((1, -1))
        pts_x = pts_x / numpy.array(scale_axes).reshape((1, -1))
    shape = (len(pts_x), len(pts))
    
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
    w = non_isotropic_dist(pts_x, pts, idx, distance_func=distance_func,
                           directionality_axis=directionality_axis,
                           directionality_fac=directionality_fac)
    if custom_w_in is not None and custom_w_out is not None:
        cust_w = custom_weight_evaluation(custom_w_out, custom_w_in, idx)
        cust_w = cust_w / cust_w.mean()
        w = w * cust_w
        print("Used custom weights!")
    
    
    if p_pick is not None:
        assert p_pick <= 1.0
        wmean = w.mean()
        picker_func = lambda _x: _x.loc[numpy.random.rand(len(_x)) < (p_pick * _x / wmean)]
    elif n_pick is not None:
        picker_func = lambda _x: _x.iloc[numpy.random.choice(len(_x), numpy.minimum(n_pick, len(_x)),
                                                             replace=False,
                                                             p=_x / _x.sum())]
    w = w.groupby("neuron").apply(picker_func).droplevel(0)
    return to_csc_matrix(w, mirror=mirror, shape=shape)
