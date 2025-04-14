import numpy
import pandas
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree


dbins = numpy.linspace(0, 1000, 30)
dbins2d = numpy.linspace(-500, 500, 31)


def calc_delta_matrix(col):
    return col.reshape((-1, 1)) - col.reshape((1, -1))

class DDtest(object):
    def __init__(self, instance, cols=["x", "y", "z"]):
        self.m = instance
        pts = self.m.vertices[cols].values
        self.pts = pts

        self.d = squareform(pdist(pts))
        self.deltas = [
            calc_delta_matrix(pts[:, a]) for a in range(pts.shape[1])
        ]
        self.h_all = numpy.histogram(self.d.flatten(), bins=dbins)[0]
        self.h_2d_all = dict([
            ((i, j),
            numpy.histogram2d(
                self.deltas[i].flatten(), self.deltas[j].flatten(),
                bins=(dbins2d, dbins2d)
            )[0])
            for i, j in [(0, 1), (0, 2), (1, 2)]
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
        H = numpy.histogram2d(
            self.deltas[i][coo.row, coo.col],
            self.deltas[j][coo.row, coo.col],
            bins=(dbins2d, dbins2d))[0]
        return H / self.h_2d_all[ij]
    
    def plot_degree_distribution(self, mdl_instance, nbins=71):
        from matplotlib import pyplot as plt

        reference = self.m.matrix.astype(bool)
        mdl_instance = mdl_instance.astype(bool)

        df_degrees = pandas.DataFrame({
            "indegree_model": numpy.array(mdl_instance.sum(axis=0))[0],
            "indegree_ref": numpy.array(reference.sum(axis=0))[0],
            "outdegree_model": numpy.array(mdl_instance.sum(axis=1))[:, 0],
            "outdegree_ref": numpy.array(reference.sum(axis=1))[:, 0]
        })

        print(df_degrees.mean())
        mx_degree = df_degrees.apply(lambda _x: numpy.percentile(_x, 99)).max()
        deg_bins = numpy.linspace(0, mx_degree, nbins)
        df_dgtz = df_degrees.apply(lambda _x: numpy.digitize(_x, bins=deg_bins) - 1)
        df_count = df_dgtz.apply(lambda _x: _x.value_counts().sort_index())

        fig = plt.figure()

        ax = fig.gca()

        ax.plot(deg_bins[df_count.index], df_count["indegree_model"], marker='x', lw=0.5, color="blue", label="Indeg. model")
        ax.plot(deg_bins[df_count.index], df_count["indegree_ref"], marker='o', lw=0.5, color="teal", label="Indeg. Ref.")
        ax.plot(deg_bins[df_count.index], df_count["outdegree_model"], marker='x', lw=0.5, color="orange", label="Outdeg. model")
        ax.plot(deg_bins[df_count.index], df_count["outdegree_ref"], marker='o', lw=0.5, color="red", label="Outdeg. ref")

        ax.set_xlabel("Degree")
        ax.set_ylabel("Neuron count")
        ax.set_yscale("log")
        plt.legend()

    def dist_and_nn_analysis(self, matrix, direction="efferent"):
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
        return H_x_con / H_x_all  # Probability that a neuron with connected NN is connected itself (per dist bin)

    @staticmethod
    def simplex_counts_and_controls(matrix):
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
    
    def plot_simplex_counts(self, matrix):
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
