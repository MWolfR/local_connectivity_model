import numpy
import pandas
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from scipy import stats, sparse

from matplotlib import pyplot as plt


dbins = numpy.linspace(0, 1000, 30)
dbins2d = numpy.linspace(-500, 500, 31)


def calc_delta_matrix(col):
    return col.reshape((-1, 1)) - col.reshape((1, -1))

class DDtest(object):
    """
    An object that facilitates comparing a random graph to a reference with respect to a number
    of graph measurements.

    NOTE: This class is very much optimized for use with the specific references connectomes we 
    considered in the manuscript and may require changes for other connectomes.
    """
    def __init__(self, instance, cols=["x", "y", "z"], max_2d_offset=50.0):
        """
        Args:
          instance (conntility.ConnectivityMatrix): The reference connectome

          cols (list): The name of the node properties of the reference to use for distance calculations.
        """
        self.m = instance
        self._max_2d_offset = max_2d_offset
        pts = self.m.vertices[cols].values
        self.pts = pts

        self.d = squareform(pdist(pts))
        self.deltas = [  # offsets along all axes
            calc_delta_matrix(pts[:, a]) for a in range(pts.shape[1])
        ]
        self.h_all = numpy.histogram(self.d.flatten(), bins=dbins)[0]
        self.h_2d_all = dict([
            ((i, j),
            numpy.histogram2d(
                self.deltas[i].flatten()[numpy.abs(self.deltas[k].flatten()) < self._max_2d_offset],
                self.deltas[j].flatten()[numpy.abs(self.deltas[k].flatten()) < self._max_2d_offset],
                bins=(dbins2d, dbins2d)
            )[0])
            for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        ])

        self.p_ref = self.for_matrix(self.m.matrix)
        self.p_2d_ref = dict([(ij, self.twod_for_matrix(self.m.matrix, ij))
                              for ij in [(0, 1), (0, 2), (1, 2)]])

    def for_matrix(self, m):
        coo = m.tocoo()
        H = numpy.histogram(self.d[coo.row, coo.col], bins=dbins)[0]
        return H / self.h_all
    
    def twod_for_matrix(self, m, ij):
        coo = m.tocoo()
        assert ij in self.h_2d_all
        i, j = ij
        k = [_dim for _dim in [0, 1, 2] if _dim not in ij][0]
        mask = numpy.abs(self.deltas[k][coo.row, coo.col]) < self._max_2d_offset
        H = numpy.histogram2d(
            self.deltas[i][coo.row, coo.col][mask],
            self.deltas[j][coo.row, coo.col][mask],
            bins=(dbins2d, dbins2d))[0]
        return H / self.h_2d_all[ij]

    def degree_distribution_analysis(self, mdl_sources, mdl_sources_names):
        """
        Fig. 2D1
        """
        n_deg_bins = 31

        def node_inout_degrees(m, pts):
            m = m.astype(bool)
            df_degrees = pandas.DataFrame({
                "indegree": numpy.array(m.sum(axis=0))[0],
                "outdegree": numpy.array(m.sum(axis=1))[:, 0]
            })
            return df_degrees

        def fit_log_distribution(x):
            if not hasattr(x, "__len__"):
                return numpy.nan
            logx = numpy.log10(x)
            valid = ~numpy.isnan(logx) & ~numpy.isinf(logx)
            return numpy.polyfit(deg_bins[:-1][valid], logx[valid], 1)[0]

        deg_types = ["indegree", "outdegree"]
        deg_analysis = {
            "analyses": {
                "degrees":{
                    "source": node_inout_degrees,
                    "output": "DataFrame"
                }
            }
        }
        deg_res = pandas.concat(
            [
                self.m.analyze(deg_analysis)["degrees"]
            ] + [
                _src.analyze(deg_analysis)["degrees"].reset_index(0)
                for _src in mdl_sources
            ],
            axis=0, keys=["reference"] + mdl_sources_names, names=["source"]
        ).fillna(0)
        mx_deg = numpy.percentile(deg_res["outdegree"].loc["reference"], 99)
        deg_bins = numpy.linspace(0, mx_deg, n_deg_bins)

        plt_kwargs = {
            "reference": {"color": "black", "lw": 1.5, "marker": "o", "ms": 5},
            mdl_sources_names[0]: {"color": "red", "lw": 0.5, "marker": "o", "ms": 3},
            mdl_sources_names[1]: {"color": "teal", "lw": 0.5, "marker": "o", "ms": 3},
            mdl_sources_names[2]: {"color": "rebeccapurple", "lw": 0.5, "marker": "o", "ms": 3}
        }

        fig = plt.figure(figsize=(5, 2.2))

        fit_res = []
        for i, col_use in enumerate(deg_types):
            Hs = deg_res.groupby(["source", "instance"])[col_use].apply(lambda _x: numpy.histogram(_x, bins=deg_bins)[0]).unstack("source")
            fit_res.append(Hs.applymap(fit_log_distribution))

            ax = fig.add_subplot(1, 2, i + 1)

            for src in Hs.columns:
                H = Hs[src]
                for j, data in H.items():
                    lbl = None
                    if j == 0:
                        lbl = src
                    if hasattr(data, "__len__"):
                        ax.plot(deg_bins[:-1], data, **plt_kwargs[src], label=lbl)
            ax.set_yscale("log")
            ax.set_xlabel(col_use)
            if i == 0: 
                ax.set_ylabel("count")
                plt.legend()
            ax.set_frame_on(False)

        fit_res = pandas.concat(fit_res, axis=1, keys=deg_types, names=["degree type"])
        fig2 = plt.figure(figsize=(1., 2))
        ax = fig2.gca()
        o = 0
        for i, col_use in enumerate(deg_types):
            fit_res_type = fit_res[col_use]
            for src in fit_res_type.columns:
                if src == "control":
                    continue
                ax.bar(o, fit_res_type[src].mean(), color=plt_kwargs[src]["color"])
                ax.errorbar(o, fit_res_type[src].mean(), yerr=fit_res_type[src].std(), color=plt_kwargs[src]["color"])
                o += 1
        ax.set_xticks(numpy.arange(len(deg_types)) * 2 + 0.5); ax.set_xticklabels(deg_types, rotation="vertical")
        ax.set_ylabel("Slope")
        ax.set_frame_on(False)
        return fig, fig2

    def dist_and_nn_analysis(self, matrix, direction="efferent"):
        """
        Fig. 2D
        """
        if direction == "efferent":
            O = matrix.tocsr()
        elif direction == "afferent":
            O = matrix.tocsc()

        kd = KDTree(self.pts)
        _, idx = kd.query(self.pts, numpy.arange(2, 3))
        idx = idx[:, 0]

        dists_x_all = []; dists_x_con = []
        for i, ab in enumerate(zip(O.indptr[:-1], O.indptr[1:])): # per row / col
            a, b = ab
            __idx = O.indices[a:b]  # indices of connected neurons (to /from the neuron of this row / col)
            nn_idx = idx[__idx]  # indices of neurons that have their nearest neighbor being connected
            is_con = numpy.isin(nn_idx, __idx)  # which neurons with connected NN are connected themselves?
            _d = self.d[i, nn_idx]
            dists_x_all.extend(_d)  # distances of neurons that have their nn connected
            dists_x_con.extend(_d[is_con])  # distances of neurons that have their nn connected and themselves connected
        H_x_all = numpy.histogram(dists_x_all, bins=dbins)[0]
        H_x_con = numpy.histogram(dists_x_con, bins=dbins)[0]
        
        ret = H_x_con / H_x_all  # Probability that a neuron with connected NN is connected itself (per dist bin)
        ret[H_x_all < 10] = numpy.nan  # Where insufficient num samples: make nan
        return ret

    @staticmethod
    def simplex_counts_and_controls(matrix):
        """
        Utility function for: Fig. 2E
        """
        import connalysis
        O = matrix.tocsc()
        keys = []
        res = []

        _N = connalysis.randomization.ER_shuffle(O)
        keys.append("ER")
        res.append(connalysis.network.simplex_counts(_N.astype(bool), max_dim=10))

        _N = connalysis.randomization.configuration_model(O)
        keys.append("Config. model")
        res.append(connalysis.network.simplex_counts(_N.astype(bool), max_dim=10))
        
        _N = connalysis.randomization.bishuffled_model(O)
        keys.append("Bishuffled model")
        res.append(connalysis.network.simplex_counts(_N.astype(bool), max_dim=10))

        keys.append("Original")
        res.append(connalysis.network.simplex_counts(O, max_dim=10))
        return pandas.concat(res, axis=0, keys=keys)
    
    @staticmethod
    def simplex_counts_over_instances(instances):
        """
        Utility function for: Fig. 2E
        """
        all_smplx_model = pandas.concat([
            DDtest.simplex_counts_and_controls(_instance)
            for _instance in instances
        ], axis=1, names=["instance"], keys=range(len(instances)))
        return all_smplx_model
    
    def simplex_count_analysis(self, src_instances, src_labels):
        """
        Utility function for: Fig. 2E
        """
        res_reference = self.simplex_counts_and_controls(self.m.matrix)
        res_reference = pandas.concat([res_reference], axis=1)
        res_srcs = [
            self.simplex_counts_over_instances(_src)
            for _src in src_instances
        ]
        fig = plt.figure(figsize=(11, 2))
        axes = fig.subplots(1, 4, sharey=True, sharex=True)

        cols = {"Original": "black", "ER": "red", "Config. model": "orange", "Bishuffled model": "purple"}

        for src, ttl, ax in zip([res_reference] + res_srcs,
                                ["reference"] + src_labels,
                                axes):
            for i in src.columns:
                smplx_model = src[i].unstack(fill_value=0)
                for mdl in smplx_model.index:
                    ax.plot(smplx_model.loc[mdl], color=cols[mdl], marker='o', ms=3, lw=0.5)

            smplx_model = src.mean(axis=1).unstack(fill_value=0)
            for mdl in smplx_model.index:
                ax.plot(smplx_model.loc[mdl], label=mdl, color=cols[mdl], lw=2.0)

            ax.set_title(ttl)
            ax.set_xticks(numpy.arange(8))
            ax.set_xlabel("Dimension")
            if ttl == "reference":
                ax.set_ylabel("Simplex count")
            ax.set_frame_on(False)
            plt.legend()
        return fig

    
    def plot_simplex_counts(self, matrix):
        """
        Fig. 2E
        """
        from matplotlib import pyplot as plt

        smplx_model = self.simplex_counts_and_controls(matrix).unstack(fill_value=0)
        smplx_ref = self.simplex_counts_and_controls(self.m.matrix).unstack(fill_value=0)

        fig = plt.figure(figsize=(7, 3))

        cols = {"Original": "black", "ER": "red", "Config. model": "orange", "Bishuffled model": "purple"}

        ax = fig.add_subplot(1, 2, 1)
        for mdl in smplx_model.index:
            ax.plot(smplx_model.loc[mdl], label=mdl, color=cols[mdl], marker='o')
        ax.set_title("Model")
        ax.set_xlabel("Dimension"); ax.set_ylabel("Count")
        plt.legend()

        ax = fig.add_subplot(1, 2, 2, sharey=ax)
        for mdl in smplx_ref.index:
            ax.plot(smplx_ref.loc[mdl], label=mdl, color=cols[mdl], marker='o')
        ax.set_title("Reference")
        ax.set_xlabel("Dimension"); ax.set_ylabel("Count")

def nnz(m, pts):
    """
    Number of nonzero elements of a matrix / edges of a graph.
    """
    return m.nnz

def mean_degree(m, pts):
    """
    Mean degree of nodes in a graph.
    """
    n = m.shape[0]
    return nnz(m, pts) / n

def _cn_bias(m, pts, direction):
    """
    Strength of common neighbor bias, as defined in the manuscript Methods,
    """
    if m.nnz == 0:
        return numpy.NaN
    m = m.astype(bool)
    adj = m.astype(float)
    assert adj.shape[0] == adj.shape[1], "Matrix must be square for this analysis!"
    nelem = adj.shape[0] * adj.shape[1]
    ndiag = adj.shape[0]
    ncon = adj.nnz
    nuncon = nelem - ndiag - ncon

    if direction == "efferent":
        cn = adj * adj.transpose()
    elif direction == "afferent":
        cn = adj.transpose() * adj
    else:
        raise ValueError("direction must be 'efferent' or 'afferent'")
    mn_elem = cn.mean()
    mn_diag = cn[numpy.diag_indices_from(cn)].mean()
    mn_con = cn[m].mean()
    mn_uncon = (mn_elem * nelem - mn_diag * ndiag - mn_con * ncon) / nuncon

    return mn_con / mn_uncon

def cn_bias_aff(m, pts):
    """
    Strength of afferent common neighbor bias. Not used in the manuscript.
    """
    return _cn_bias(m, pts, "afferent")

def cn_bias_eff(m, pts):
    """
    Strength of efferent common neighbor bias. This one's used in the manuscript.
    """
    return _cn_bias(m, pts, "efferent")

def _skewness_deg_dist_fit(m, pts, direction):
    """
    Skewness of a lognormal distribution fit to the degree distribution of a graph.
    """
    m = m.tocoo()
    if m.nnz == 0:
        return numpy.NaN
    if direction == "efferent":
        degs = pandas.Series(m.row).value_counts()
    elif direction == "afferent":
        degs = pandas.Series(m.col).value_counts()

    d_shape, d_loc, d_scale = stats.lognorm.fit(degs)
    return (numpy.exp(d_shape ** 2) + 2) * numpy.sqrt(numpy.exp(d_shape ** 2) - 1)

def _skewness_deg_dist_samples(m, pts, direction):
    """
    Skewness of the degree distribution of a graph as scipy.stats calculates it.
    """
    m = m.tocoo()
    if m.nnz == 0:
        return numpy.NaN
    if direction == "efferent":
        degs = pandas.Series(m.row).value_counts()
    elif direction == "afferent":
        degs = pandas.Series(m.col).value_counts()

    return stats.skew(degs.values)

def skewness_deg_dist_eff(m, pts):
    """
    Skewness of the out-degree distribution of a graph.
    """
    return _skewness_deg_dist_samples(m, pts, "efferent")

def skewness_deg_dist_aff(m, pts):
    """
    Skewness of the in-degree distribution of a graph.
    """
    return _skewness_deg_dist_samples(m, pts, "afferent")

def con_prob_within(m, pts, max_dist=50.0):
    """
    Connection probability within the specifed distance for a graph where nodes
    are associated with x, y, z coordinates.
    """
    m = m.tocsr().astype(bool)
    pts = pts[["x", "y", "z"]].values
    tree = KDTree(pts)
    idx = tree.query_ball_point(pts, max_dist)
    idx = [_idx[1:] for _idx in idx]
    l = numpy.cumsum([0] + list(map(len, idx)))
    idx_stack = numpy.hstack(idx)
    m_idx = sparse.csr_matrix((numpy.ones(len(idx_stack), dtype=bool), idx_stack, l), shape=(len(pts), len(pts)))
    return m[m_idx].mean()
