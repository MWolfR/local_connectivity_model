import numpy
import pandas

from scipy import stats, sparse
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


def to_csc_matrix(w, mirror=True, shape=None):
    """
    Transforms an intermediate representation of a random geometric graph to a scipy.sparse.csc_matrix.
    Only used internally.
    """
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


def non_isotropic_dist(pts_x, pts, idx, directionality_fac=0, directionality_axis=None, distance_func=None):
    """
    Calculates weights for picking edges of a random geometric graph based (mostly) on directionality, i.e.,
    how well the direction of the vector from the source to the target node aligns with a specified axis.
    Note that the potential nodes that can be connected to is calculated separately and is an input.

    Args:
      pts_x (numpy.array, m X 3): Locations in space of the potential source nodes.
      
      pts (numpy.array, m X 3): Locations in space of the potential target nodes. Note: In the manuscript we
      only explore intrinsic connectivity, i.e., the case pts_x == pts. But here, we are prepared to also use
      the function for connections from one population to another.
      
      idx (numpy.array of lists of ints): Indices of nodes to potentially connect to. 
      One entry per node; the entry is a list of the indices of other nodes within distance d, i.e., that are
      close enough to connect to.

      directionality_fac (float): How strong the bias for edges in a specific direction is. w_A of the manuscript.

      directionality_axis (numpy.array, shape=(3,)): The axis defining the prefered direction. A of the manuscript.

      distance_func: Only used if no directionality (A and w_A) are specified. A function that yields values of
      weights based on distance, such as exp(-distance). Not used in the manuscript.
    """
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
    """
    Generates "per node bias" weights based on the strengths of pathways in a reference connectome to match.
    For details, see the "per node bias" subsection in the Methods of the manuscript.

    Args:
      reference (conntility.ConnectivityMatrix): A reference connectome whose pathway strengths we want to match.
      
      property_to_use (str): The name of a node property that must exist in the reference connectome above. 
      Defines the pathways to match: Each pathway if a combination of possible values of the node property for
      source and target nodes. Essentially: Neuron types.
      
      axis (0 to 1): If 0: Calculates biases for nodes as target nodes (w_i in the manuscript). If 1: same, but
      as source nodes (w_o in the manuscript).
    """
    c = reference.vertices[property_to_use].value_counts().sort_index()
    nrml = c.values.reshape((-1, 1)) * c.values.reshape((1, -1)) + 1E-9 # Number of pairs, avoiding div. by 0

    MM = reference.condense(property_to_use).array
    MM = MM / nrml  # Connection prob
    MM = MM.mean() * (MM ** 1.5) / (MM ** 1.5).mean()
    w_per_class = numpy.nanmean(MM, axis=axis) / numpy.sqrt(numpy.nanmean(MM))
    w_per_class = pandas.Series(w_per_class, name="weight", index=c.index)
    w_per_node = w_per_class[reference.vertices[property_to_use]].values
    return w_per_node


def custom_weight_evaluation(w_out, w_in, idx):
    """
    Evaluates the "per node bias" weights (w_o and w_i in the manuscript) to yield one values for
    each potential edge of the random geometric graph.

    Args:
      w_out (numpy.array): w_o of the manuscript. One entry per node.
      
      w_in (numpy.array): w_i of the manuscript. One entry per node.

      idx (numpy.array of lists of ints): Indices of nodes to potentially connect to. 
      One entry per node; the entry is a list of the indices of other nodes within distance d, i.e., that are
      close enough to connect to.
    """
    w_out = numpy.array(w_out)
    w_in = numpy.array(w_in)

    W = [pandas.Series(_w_o * w_in[_idx], name="weight",
                       index=pandas.Index(_idx, name="i"))
         for _w_o, _idx in zip(w_out, idx)]
    W = pandas.concat(W, axis=0, keys=range(len(idx)), names=["neuron"])
    return W


def cand2_point_nn_matrix(pts, pts_x=None, n_neighbors=4, dist_neighbors=None, n_pick=None,
                    p_pick=None, scale_axes=None,
                    directionality_fac=0.0, directionality_axis=None,
                    distance_func=None,
                    custom_w_out=None, custom_w_in=None,
                    no_diag=True, mirror=False):
    """
    Generates a random geometric graph with some modifications outlined in the manuscript.

    Args:
      pts (numpy.array, m X 3): Locations in space of the nodes.
      
      pts_x (optional, numpy.array, m X 3): If specified, pts_x is the locations of source nodes and pts the
      locations of target nodes of the random geometric graph. If not specified, pts_x = pts. In the manuscript we
      only explore intrinsic connectivity, i.e., the case pts_x == pts. But here, we are prepared to also use
      the function for connections from one population to another.

      n_neighbors (optional, int): One way of specifying the potential nodes to connect each node to. If given,
      each node is potentially connected to the specified number of nearest neighbors. This method is NOT used
      in the manuscript.

      dist_neighbors (optional, float): The alternative way of specifying potential nodes. If given, each node 
      potentially connected to nodes withing the specified distance. IF BOTH n_neighbors AND dist_neighbors IS
      GIVEN, THE BEHAVIOR IS UNDEFINED!

      n_pick (optional, int): One way of specifying the sparsity of the graph. Each node will be connected to 
      exactly that number of other nodes, unless the number of potential targets is smaller than that. This method
      is NOT used in the manuscript.

      p_pick (optional, float): The alternative way of specifying the sparsity. Each node is connected to each of
      its potential partners with that probability. IF BOTH n_pick AND p_pick IS GIVEN, THE BEHAVIOR IS UNDEFINED!

      scale_axes (optional, numpy.array): Length must match the second dimension of pts and pts_x. Divides each
      coordinate of the points by the corresponding value.

      directionality_fac (float): How strong the bias for edges in a specific direction is. w_A of the manuscript.

      directionality_axis (numpy.array, shape=(3,)): The axis defining the prefered direction. A of the manuscript.

      distance_func: Only used if no directionality (A and w_A) are specified. A function that yields values of
      weights based on distance, such as exp(-distance). Not used in the manuscript.

      custom_w_out (numpy.array): Length must match length of pts_x. "per node bias" weights, i.e., w_o
      of the manuscript. 

      custom_w_out (numpy.array): Length must match length of pts. "per node bias" weights, i.e., w_i
      of the manuscript. 

      no_diag (bool; default: True): If set to False, connections from a node to itself are allowed. NOT used
      in the manuscript.

      mirror (bool; default: False): If set to True, the output matrix is made symmetrical. That is, if an edge
      exists from a to b, the edge from b to a is also added, if it does not already exist. NOT used in the
      manuscript.
    """
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
    w = w.groupby("neuron").apply(picker_func)
    if len(w) > 0:
        w = w.droplevel(0)
        return to_csc_matrix(w, mirror=mirror, shape=shape)
    return sparse.csc_matrix((len(pts), len(pts)), dtype=bool)
