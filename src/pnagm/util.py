import numpy


# For the random or non-random generation of neuron locations in space
def make_points(cfg):
    n_nrn = cfg["n_nrn"]
    tgt_sz = cfg["tgt_sz"] 
    pts = numpy.random.rand(n_nrn, 3) * tgt_sz - tgt_sz/2
    return pts

def points_from_microns(cfg):
    import conntility
    fn = cfg["fn"]
    N = conntility.ConnectivityMatrix.from_h5(fn, "condensed")
    sz = cfg["tgt_sz"]
    cols = ["x_nm", "y_nm", "z_nm"]
    for _col in cols:
        N._vertex_properties[_col[0]] = N._vertex_properties[_col] / 1000.0
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
    return pts, N
