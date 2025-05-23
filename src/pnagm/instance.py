import numpy
import pandas

from scipy import sparse

def _evaluate_probs_less_random(p_mat, adjust=None):
    """
    Evaluates a single spread step with the "less stochastic" modification in place.
    Args:
      p_mat (scipy.sparse matrix): Represents the current state of the spread. Each row corresponds to
      a node to spread from. Columns that we can spread to have positive entries, columns that we already
      spread to or that are excluded on account of being candidates from earlier steps have negative values.
      All others are zero.
      
      adjust (optional, numpy.array): Each entry is 1/r of the corresponding node. See manuscript for what "r" means.
    """
    p_mat = p_mat.tocsr()
    n_pick = numpy.array(p_mat.sum(axis=1))[:, 0]
    if adjust is not None:
        n_pick = n_pick * adjust
    n_pick = numpy.round(n_pick).astype(int)

    indptr_out = [0]
    picked = []
    for a, b, c in zip(p_mat.indptr[:-1], p_mat.indptr[1:], n_pick):
        p = p_mat.data[a:b] / p_mat.data[a:b].sum()
        c = numpy.minimum(c, (p > 0).sum())
        indptr_out.append(c)
        if c > 0:
            picked.append(numpy.random.choice(p_mat.indices[a:b], c, p=p, replace=False))
    picked = numpy.hstack(picked)
    indptr_out = numpy.cumsum(indptr_out)
    m_out = sparse.csr_matrix((numpy.ones(indptr_out[-1], dtype=bool), 
                            numpy.hstack(picked),
                            indptr_out),
                            shape=p_mat.shape).tocoo()
    return m_out

# For wiring up the model, based on the nearest neighbor graph.
def evaluate_probs(p_mat, adjust=None, less_random=False):
    """
    Evaluates a single spread step.
    Args:
      p_mat (scipy.sparse matrix): Represents the current state of the spread. Each row corresponds to
      a node to spread from. Columns that we can spread to have positive entries, columns that we already
      spread to or that are excluded on account of being candidates from earlier steps have negative values.
      All others are zero.
      
      adjust (optional, numpy.array): Each entry is 1/r of the corresponding node. See manuscript for what "r" means.
      
      less_random (optional, bool; default: False): If set to True, uses the "less stochastic" modification 
      desribed in the manuscript
    """
    if less_random:
        return _evaluate_probs_less_random(p_mat, adjust=adjust)
    p_mat = p_mat.tocoo()
    thresh = p_mat.data
    if adjust is not None:
        thresh = thresh * adjust[p_mat.row]
    _v = numpy.random.rand(p_mat.nnz) < thresh
    return sparse.coo_matrix((numpy.ones(_v.sum()), (p_mat.row[_v], p_mat.col[_v])), shape=p_mat.shape)

def build_instance(pts, M, n_steps=100,
                   sum_exclusion=True, step_tgt=10, n_protected=0,
                   fac_protected=1.0,
                   tgt_level="individual", decay=1.0):
    """
    Builds a stochastic spread graph.
    Args:
      pts (numpy.array; m X n): The locations of the nodes. Note: Locations are not actually used at this stage. We only consider its lengths
      to understand the number of nodes.

      M (scipy.sparse matrix. m X m): The underlying graph to spread on.

      n_steps (default: 100): Max. number of spread steps to evaluate.

      sum_exclusion (default: True): If set to False, then only candidate nodes from the previous step
      are excluded from the next step instead of cumulatively growing the set. That is, it determines the
      update of T_i from step to step. This function is NOT USED in the manuscript.

      step_tgt: Parameter q of the manuscript.

      n_protected (default: 0): Parameter k of the manuscript.

      fac_protected (default: 1.0): A multiplier applied to q during the first k steps. Not used, i.e., set to 1.0
      in the manuscript.

      tgt_level (str, one of ["mean", "individual"]; default: "individual"): If set to "mean", then only a single
      value of r is calculated for all nodes, otherwise one per node is used. In the manuscript, we only ever used
      "individual". Refer to the manuscript for an explanation of r.

      decay (default: 1.0): At each step, q is multiplied with this value to generate a new value of q, thereby decaying
      its value over time. Not used (i.e., set to 1.0) in the manuscript.
    """
    sum_exclusion = bool(sum_exclusion)
    exclusion = sparse.coo_matrix(([], ([], [])), shape=M.shape)

    initial = sparse.coo_matrix((numpy.ones(len(pts)),
                                (numpy.arange(len(pts)), numpy.arange(len(pts)))),
                                shape=M.shape).tocsr()

    row = []
    col = []
    data = []

    state = initial
    history = []
    M = M.transpose()

    for _step in range(n_steps):
        candidates = state * M - 100 * (exclusion + initial)
        candidates.data = numpy.minimum(numpy.maximum(candidates.data, 0), 1.0)
        if step_tgt is not None:
            if tgt_level == "mean":
                csum = numpy.array(candidates.sum(axis=1))[:, 0]
                fac = step_tgt * numpy.ones(M.shape[0]) / csum[csum > 0].mean()
            else:
                fac = step_tgt / numpy.array(candidates.sum(axis=1) + 1E-3)[:, 0]
            step_tgt = step_tgt * decay
        else:
            fac = None
        if (_step < n_protected):
            candidates = fac_protected * candidates
            
        new_state = evaluate_probs(candidates, adjust=fac, less_random=(_step < n_protected))

        row.extend(new_state.row); col.extend(new_state.col); data.extend(_step * numpy.ones(new_state.nnz, dtype=int))
        new_state = new_state.tocsr()

        h_i = new_state.sum(axis=1).mean()  # Mean number added per original neuron
        history.append(h_i)
        if sum_exclusion:
            exclusion = exclusion + state
        else:
            exclusion = state
        state = new_state


    full_instance = sparse.coo_matrix((
        numpy.ones(len(row), dtype=bool),
        (row, col)
    ), shape=M.shape).tocsr()

    degs = numpy.array(full_instance.sum(axis=1))[:, 0]
    degs = pandas.DataFrame(degs).transpose()
    return full_instance, history, degs
