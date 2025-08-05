import numpy
import h5py
import pandas
import os


# For the random or non-random generation of neuron locations in space
def make_points(cfg):
    """
    Generate a random point cloude from a parameterization dict.
    """
    n_nrn = cfg["n_nrn"]
    tgt_sz = cfg["tgt_sz"]
    if not hasattr(tgt_sz, "__iter__"):
        tgt_sz = [tgt_sz, tgt_sz, tgt_sz]
    tgt_sz = numpy.array(tgt_sz).reshape((1, -1))
    pts = numpy.random.rand(n_nrn, 3) * tgt_sz - tgt_sz/2
    return pts

def no_categorical_dtypes(N):
    """
    Utility function: Turns categorical node properties into non-categorical ones. This is required
    for some analyses for technical reasons.
    """
    import pandas

    for col in N._vertex_properties.columns:
        if isinstance(N._vertex_properties[col].dtype, pandas.CategoricalDtype):
            N._vertex_properties[col] = N._vertex_properties[col].astype(str)

def load_all_instances_from_file(fn):
    """
    Utility function to load all ConnectivityMatrix objects from an .h5 file. This function is not very
    clever and strongly assumes that any group found in the .h5 file contains a ConnectivityMatrix, without
    additional checks.
    """
    import conntility

    grp_tuples = []
    with h5py.File(fn, "r") as h5:
        for prefix in h5.keys():
            if isinstance(h5[prefix], h5py.Group):
                for grp_name in h5[prefix].keys():
                    if isinstance(h5[prefix][grp_name], h5py.Group):
                        grp_tuples.append((prefix, grp_name))
    matrices = [conntility.ConnectivityMatrix.from_h5(fn, group_name=grp_name,
                                                      prefix=prefix)
                for prefix, grp_name in grp_tuples]
    matrices = conntility.ConnectivityGroup(
        pandas.DataFrame({"instance": range(len(matrices))}), matrices
    )
    return matrices

def points_from_microns(cfg, return_additional_controls=False):
    """
    Generate a point cloud by looking up soma locations from a reference connectome. The reference could
    be the MICrONS EM connectome, hence the name. But in principle any source can be used, as long as it 
    provides soma locations. 

    Args:
      cfg: dict configuring the process, i.e., which connectome to load and which subvolume to select.
    """
    import conntility

    if "root" in cfg: 
        fn = cfg["root"] + "/" + cfg["fn"]
    else: 
        fn = cfg["fn"]
    try:
        N = conntility.ConnectivityMatrix.from_h5(fn, "condensed")
    except:
        N = conntility.ConnectivityMatrix.from_h5(fn)
    no_categorical_dtypes(N)
    
    sz = cfg["tgt_sz"]
    cols = ["x_nm", "y_nm", "z_nm"]
    for _col in cols:
        if _col in N._vertex_properties.columns:
            N._vertex_properties[_col[0]] = N._vertex_properties[_col] / 1000.0
    
    tl_col_dict = {
        "ss_flat_x": "x", "ss_flat_y": "z", "depth": "y"
    }
    for _col_in, _col_out in tl_col_dict.items():
        if _col_in in N._vertex_properties.columns:
            N._vertex_properties[_col_out] = N._vertex_properties[_col_in] + numpy.random.rand(len(N)) * 1E-9
    cols = ["x", "y", "z"]

    center = N.vertices[cols].mean()
    for k, v in cfg.get("filters", {"classification_system": "excitatory_neuron"}).items():
        if isinstance(v, list):
            N = N.index(k).isin(v)
        else:
            N = N.index(k).eq(v)

    for _col in cols:
        _col_o = "o_" + _col
        if _col_o in cfg:
            _o = cfg[_col_o]
            N = N.index(_col).le(center[_col] + _o + sz/2).index(_col).ge(center[_col] + _o - sz/2)
            print(len(N))
    
    pts = N.vertices[cols].values

    if return_additional_controls:
        additional_controls = {}
        for add_ctrl_name, add_ctrl in cfg.get("additional_controls", {}).items():
            ctrl_fn = add_ctrl["fn"]
            if "root" in add_ctrl.keys():
                ctrl_fn = os.path.join(add_ctrl["root"], ctrl_fn)
            additional_controls[add_ctrl_name] = load_all_instances_from_file(ctrl_fn)
        return pts, N, additional_controls
    return pts, N

def create_neighbor_spread_graph(pts, cfg, reference=None):
    """
    Utility function that combines generation of a random geometric graph and generation of a stochastic
    spread graph on the first graph. 

    Args:
      pts (numpy.array; m x n): Locations of m nodes in n-dimensional space.

      cfg (dict): Configures generation of the two graphs. For details, see README and the docstrings of
      pnagm.nngraph.cand2_point_nn_matrix and instance.build_instance.

      reference (optional; conntility.ConnectivityMatrix): If "per_class_bias" is used for the generation of
      the random geometric graph, then a reference connectome is required. See README and 
      nngraph.generate_custom_weights_by_node_class for details.
    """
    from . import nngraph, instance

    w_out_use = None
    w_in_use = None
    if "per_class_bias" in cfg:
        assert reference is not None, "When using per class bias, must provide reference ConnectivityMatrix"
        if "outgoing" in cfg["per_class_bias"]:
            prop = cfg["per_class_bias"]["outgoing"]
            w_out_use = nngraph.generate_custom_weights_by_node_class(reference, prop, 1)
        if "incoming" in cfg["per_class_bias"]:
            prop = cfg["per_class_bias"]["incoming"]
            w_in_use = nngraph.generate_custom_weights_by_node_class(reference, prop, 0)
        
    M = nngraph.cand2_point_nn_matrix(pts,
                                      custom_w_out=w_out_use, custom_w_in=w_in_use,
                                      **cfg["nngraph"]).astype(bool).astype(float)
    mdl_instance, a, b = instance.build_instance(pts, M, **cfg["instance"])

    return mdl_instance, M
 