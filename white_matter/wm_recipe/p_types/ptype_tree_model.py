import numpy
import networkx as nx


def map_step(p1, p2):
    u = numpy.unique(p1)
    splt = [p2[p1 == _u] for _u in u]
    counter = [numpy.unique(_s, return_counts=True)
               for _s in splt]
    counter = [_x[0][numpy.argsort(_x[1])].tolist()
               for _x in counter]
    idx = numpy.max(u) + 1
    mp = {}
    for _c, _u in zip(counter, u):
        n = _c.pop()
        if n not in mp:
            mp[n] = _u
    while(True):
        if numpy.sum(map(len, counter)) == 0:
            break
        for _c, _u in zip(counter, u):
            if len(_c) == 0:
                continue
            n = _c.pop()
            if n not in mp:
                mp[n] = idx
                idx += 1
    return mp


def adjust_groups(rr, P):
    valid = numpy.nonzero(numpy.diff(P, axis=0).sum(axis=1))[0] + 1
    P = P[numpy.hstack([0, valid])]
    rr = rr[numpy.hstack([0, valid])]
    for i in range(len(P) - 1):
        mp = map_step(P[i], P[i + 1])
        for j in range(P.shape[1]):
            P[i + 1, j] = mp[P[i + 1, j]]
    return rr, P


def make_tree(rr, P):
    def group_contents(lst, p, G):
        return [[p[i] for i in G.nodes[g]['contents']]
                for g in lst]

    # noinspection PyUnresolvedReferences
    def merge_list(lst, p, G):
        groups = group_contents(lst, p, G)
        merge_groups = numpy.arange(len(groups))
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups[i + 1:]):
                if numpy.in1d(g1, g2).sum() > (0.5 * len(g1)) and\
                   numpy.in1d(g2, g1).sum() > (0.5 * len(g2)):
                    merge_groups[merge_groups == merge_groups[i + 1 + j]] = merge_groups[i]
        merge_sets = [[l for grp, l in zip(merge_groups, lst) if grp == u_grp]
                      for u_grp in numpy.unique(merge_groups) if numpy.sum(merge_groups == u_grp) > 1]
        return merge_sets

    def merge_step(lst, p, r, G, idx):
        to_merge = merge_list(lst, p, G)
        for m in to_merge:
            for _m in m:
                lst.remove(_m)
            n = [G.nodes[_m] for _m in m]
            contents = numpy.unique(numpy.hstack([_n['contents'] for _n in n])).tolist()
            G.add_node(idx, born=r, contents=contents)
            for _m, _n in zip(m, n):
                G.add_edge(idx, _m, length=_n['born'] - r, type='down')
            lst.append(idx)
            idx += 1
        return idx

    G = nx.DiGraph()
    lst_nodes = []
    for i in range(P.shape[1]):
        G.add_node(i, born=rr[-1], contents=[i])
        lst_nodes.append(i)
    idx = P.shape[1]
    for p, r in zip(P[-1::-1], rr[-1::-1]):
        idx = merge_step(lst_nodes, p, r, G, idx)
    print(lst_nodes)
    return G


# noinspection PyTypeChecker,PyDefaultArgument
def layout_tree(G, root, pos_dict=None, x=0, y=[0, 10], length=['length']):
    def make_splt(suc):
        L = [len(G.nodes[_s]['contents']) for _s in suc]
        L = numpy.hstack([0, numpy.cumsum(L)]).astype(float) / numpy.sum(L)
        dy = numpy.diff(y)[0]
        return [y[0] + L[i:i + 2] * dy for i in range(len(suc))]

    if pos_dict is None:
        pos_dict = {}
    pos_dict[root] = (x, numpy.mean(y))
    suc = list(G.successors(root))
    suc = [_suc for _suc in suc if len(G.nodes[_suc]['contents']) < len(G.nodes[root]['contents'])]
    splt = make_splt(suc)
    for n, yy in zip(suc, splt):
        l = numpy.mean([G.edges[(root, n)][_l] for _l in length])
        layout_tree(G, n, pos_dict=pos_dict, x=x+l, y=yy, length=length)
    return pos_dict


# noinspection PyTypeChecker,PyDefaultArgument
def layout_radial_tree(G, root, pos_dict=None, l=0, angle=[0, 2*numpy.pi], length=['length'], bidirectional=True):
    def make_splt(suc):
        L = [len(G.nodes[_s]['contents']) for _s in suc]
        L = numpy.hstack([0, numpy.cumsum(L)]).astype(float) / numpy.sum(L)
        da = numpy.diff(angle)[0]
        return [angle[0] + L[i:i + 2] * da for i in range(len(suc))]

    def polar2cart(pl, pa):
        return (pl * numpy.cos(pa), pl * numpy.sin(pa))

    if pos_dict is None:
        pos_dict = {}
    pos_dict[root] = polar2cart(l, numpy.mean(angle))
    suc = list(G.successors(root))
    suc = [_suc for _suc in suc if len(G.nodes[_suc]['contents']) < len(G.nodes[root]['contents'])]
    splt = make_splt(suc)
    for n, splt_a in zip(suc, splt):
        el = [G.edges[(root, n)][_l] for _l in length]
        if bidirectional:
            el += [G.edges[(n, root)][_l] for _l in length]
        el = numpy.mean(el)
        layout_radial_tree(G, n, pos_dict=pos_dict, l=l+el, angle=splt_a, length=length, bidirectional=bidirectional)
    return pos_dict


def get_leaves(T):
    return sorted([_n for _n in T.nodes if T.out_degree[_n] == 1])  # 1 because we made edges bidirectional


def get_root(T):
    return sorted(T.nodes)[-1]


def get_out_edges(T, node, edge_type='down'):
    return [_e for _e in T.out_edges(node)
            if T.edges[_e]['type'] == edge_type]


def make_bidirectional(T):
    for e in T.edges:
        tmp = T.edges[e].copy()
        tmp['type'] = 'up'
        T.add_edge(e[1], e[0], **tmp)


def con_mat2cluster_tree(M, radial=True):
    import community
    gamma = numpy.linspace(0, 12.75, 2001)
    rr = 1.0 / gamma[1:-1]
    G = nx.from_numpy_array(M + M.transpose(), create_using=nx.Graph())
    partitions = [community.best_partition(G, resolution=_r) for _r in rr]
    P = numpy.vstack([numpy.array([_part[i] for i in range(M.shape[0])]) for _part in partitions])
    P = numpy.vstack([numpy.zeros(P.shape[1], dtype=int), P, numpy.arange(P.shape[1], dtype=int)])
    T = make_tree(gamma, P)
    make_bidirectional(T)
    if radial:
        pos_dict = layout_radial_tree(T, get_root(T))
    else:
        pos_dict = layout_tree(T, get_root(T))
    return T, pos_dict


def tree2dist_mat(T, weight='length'):
    leaves = get_leaves(T)
    D = [[nx.algorithms.shortest_path_length(T, i, j, weight=weight)
          for j in leaves] for i in leaves]
    return numpy.array(D)


def _get_pairs(T, node=None):
    if node is None:
        node = get_root(T)
    o_e = [_e for _e in T.out_edges(node)
           if T.edges[_e]['type'] == 'down' and 'log_p' not in T.edges[_e]]
    ret = []
    for i, e1 in enumerate(o_e):
        if 'w_out' in T.nodes[e1[1]]:
            for e2 in o_e[(i + 1):]:
                if 'w_out' in T.nodes[e2[1]]:
                    ret.append((e1[1], e2[1]))
        else:
            ret.extend(_get_pairs(T, e1[1]))
    return ret


def _merge_w(p1, p2, r, tpl_out, tpl_in, W, ND):
    ttl_w = W[p1] + W[p2]
    w_out = (W[p1] * (ND[p1, :] + tpl_out[0])
             + W[p2] * (ND[p2, :] + tpl_out[1])) / ttl_w
    w_in = (W[p1] * (ND[:, p1] + tpl_in[0])
            + W[p2] * (ND[:, p2] + tpl_in[1])) / ttl_w
    W[r] = ttl_w
    ND[r, :] = w_out
    ND[:, r] = w_in


def fit_and_merge_pair(T, pair, W, ND, L):
    N = numpy.array([[1, -1, 0, 0], [0, 0, 1, -1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
    path = nx.algorithms.shortest_path(T, pair[0], pair[1])
    assert len(path) == 3
    r = path[1]
    x_out = ND[pair[0], :L].mean() - ND[pair[1], :L].mean()
    x_in = ND[:L, pair[0]].mean() - ND[:L, pair[1]].mean()
    a1 = ND[pair[0], pair[1]]
    a2 = ND[pair[1], pair[0]]
    b = numpy.array([x_out, x_in, a1, a2])
    ir, jr, ri, rj = numpy.linalg.lstsq(N, b, rcond=None)[0]

    def _updater(e, val):
        val = numpy.maximum(val, 0.0)
        e['log_p'] = numpy.nanmean([e.get('log_p', numpy.NaN), val])
    _updater(T.edges[(pair[0], r)], -ir)
    _updater(T.edges[(pair[1], r)], -jr)
    _updater(T.edges[(r, pair[0])], -ri)
    _updater(T.edges[(r, pair[1])], -rj)
    tpl_out = (T.edges[(pair[0], r)]['log_p'], T.edges[(pair[1], r)]['log_p'])
    tpl_in = (T.edges[(r, pair[0])]['log_p'], T.edges[(r, pair[1])]['log_p'])
    _merge_w(pair[0], pair[1], r, tpl_out, tpl_in, W, ND)


def fit_tree_to_mat(T, M):
    node = get_root(T)
    L = M.shape[0]
    ND = numpy.NaN * numpy.ones((len(T.nodes), len(T.nodes)))
    ND[:L, :L] = numpy.log10(M)
    W = numpy.NaN * numpy.ones(len(T.nodes))
    W[:L] = 1
    touched = set(range(M.shape[0]))

    def _recursion(T, node):
        edges = get_out_edges(T, node)
        for i, e1 in enumerate(edges):
            if e1[1] not in touched:
                _recursion(T, e1[1])
            for e2 in edges[(i + 1):]:
                if e2[1] not in touched:
                    _recursion(T, e2[1])
                fit_and_merge_pair(T, (e1[1], e2[1]), W, ND, L)
    _recursion(T, node)
    return W, ND


class TreeInnervationModel(object):
    def __init__(self, T, p_func=lambda x: 10**-x, val_mask=None, mpr=None):
        if mpr is None:
            from white_matter.wm_recipe.parcellation import RegionMapper
            self.mpr = RegionMapper()
        else:
            self.mpr = mpr
        self.T = T
        self.p_func = p_func
        self.leaves = get_leaves(self.T)
        self._M1 = None
        if val_mask is None:
            self._val_mask = numpy.ones((len(self.leaves), len(self.leaves)), dtype=bool)
        else:
            self._val_mask = val_mask

    # noinspection PyDefaultArgument
    def grow_from(self, idx, coming_from=[], valids=None):
        if isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.mpr.region2idx(idx)
        if valids is None:
            valids = numpy.nonzero(self._val_mask[idx])[0]
        elif idx in self.leaves:
            if idx in valids:
                return [idx]
            else:
                return []
        edges = [e for e in self.T.out_edges(idx)
                 if (e[1], e[0]) not in coming_from]
        ret = []
        for e in edges:
            p = self.p_func(self.T.edges[e]['log_p'])
            if numpy.random.rand() < p:
                ret.extend(self.grow_from(e[1], coming_from=[e], valids=valids))
        return ret

    def get_interaction_strength(self, axon_from, r1, r2, weight='log_p'):
        T = self.T
        if isinstance(axon_from, str) or isinstance(axon_from, unicode):
            axon_from = self.mpr.region2idx(axon_from)
        p1 = nx.algorithms.shortest_path(T, axon_from, r1, weight=weight)
        p2 = nx.algorithms.shortest_path(T, axon_from, r2, weight=weight)
        idxx = numpy.nonzero([_p in p2 for _p in p1])[0][-1]
        dl = nx.algorithms.shortest_path_length(T, p1[idxx], r2, weight=weight) \
            - nx.algorithms.shortest_path_length(T, axon_from, r2, weight=weight)
        return self.p_func(dl)

    def interaction_mat(self, axon_from, no_redundant=False):
        T = self.T
        leaves = get_leaves(T)
        M = numpy.zeros((len(leaves), len(leaves)))
        for i, l1 in enumerate(leaves):
            if no_redundant:
                for j, l2 in enumerate(leaves[(i + 1):]):
                    M[i, j + i + 1] = self.get_interaction_strength(axon_from, l1, l2)
            else:
                for j, l2 in enumerate(leaves):
                    M[i, j] = self.get_interaction_strength(axon_from, l1, l2)
        return M

    def idx2region_hemi(self, idxx):
        if idxx > len(self.mpr.region_names):
            return (self.mpr.idx2region(idxx - len(self.mpr.region_names)), 'contra')
        return (self.mpr.idx2region(idxx), 'ipsi')

    def region_hemi_names(self):
        return [(_reg, 'ipsi') for _reg in self.mpr.region_names] +\
                [(_reg, 'contra') for _reg in self.mpr.region_names]

    def _first_order_mat(self):
        M = self.p_func(tree2dist_mat(self.T, weight='log_p'))
        M[numpy.eye(M.shape[0]) == 1] = numpy.NaN
        return M

    def first_order_mat(self):
        if self._M1 is None:
            self._M1 = self._first_order_mat()
            self._M1[~self._val_mask] = 0.0
        return self._M1

    def to_json(self, fn, overwrite=False):
        import json, os
        import networkx as nx
        if not overwrite and os.path.exists(fn):
            raise Exception("File exists: " + fn)
        with open(fn, 'w') as fid:
            json.dump(nx.node_link_data(self.T), fid)

    def draw(self, **kwargs):
        from matplotlib import pyplot as plt
        ax = plt.figure(figsize=(9, 9)).add_axes([0, 0, 1, 1])
        mpr = self.mpr
        pos = layout_radial_tree(self.T, get_root(self.T), length=['log_p'])
        lbls = dict(enumerate(mpr.region_names))
        lbls.update(dict([(i + len(mpr.region_names), v) for i, v in enumerate(mpr.region_names)]))
        cols = [[0.95, 0.5, 0.5] for _ in range(len(mpr.region_names))]
        cols.extend([[0.5, 0.5, 1.0] for _ in range(len(mpr.region_names))])
        cols.extend([[0.7, 0.7, 0.7] for _ in range(len(self.T.nodes) - 2 * len(mpr.region_names))])
        szs = [50.0] * 2 * len(mpr.region_names) + [20.0] * (len(self.T.nodes) - 2 * len(mpr.region_names))
        nx.draw_networkx(self.T, pos, font_size=8, labels=lbls, node_color=cols, node_size=szs, ax=ax, **kwargs)
        plt.axis('equal')
        plt.axis('off')

    @classmethod
    def from_con_mats(cls, mat_topology, mat_weights, optimize=False, **kwargs):
        mat_topology[numpy.isnan(mat_topology)] = 0.0 # TODO: Instead mask out
        mat_weights[numpy.isnan(mat_weights)] = 0.0
        T, pos_dict = con_mat2cluster_tree(mat_topology, radial=True)
        epsilon = mat_weights[mat_weights > 0].min()
        fit_tree_to_mat(T, mat_weights + epsilon)
        if optimize:
            for n in get_leaves(T):
                for e in T.out_edges(n):
                    T.edges[e]['log_p'] = 0.0
            mdl_tmp = cls(T)
            M1 = mdl_tmp.first_order_mat()
            M1[mat_weights == 0] = 0.0
            sbtrct = numpy.log10(numpy.polyfit(M1[~numpy.isnan(M1)],
                                               mat_weights[~numpy.isnan(M1)], 1)[0])
            for e in T.edges:
                if T.edges[e]['log_p'] > sbtrct and T.edges[e]['log_p'] > 0.0:
                    T.edges[e]['log_p'] = T.edges[e]['log_p'] - sbtrct
        return cls(T, val_mask=(mat_weights > 0), **kwargs)

    @classmethod
    def from_json(cls, fn, **kwargs):
        import networkx as nx
        import json
        with open(fn, 'r') as fid:
            data = json.load(fid)
        T = nx.node_link_graph(data)
        return cls(T, **kwargs)

    @classmethod
    def from_config(cls, cfg, **kwargs):
        import os, h5py
        if not os.path.exists(cfg["json_cache"]) or not os.path.exists(cfg["h5_cache"]):
            raise NotImplementedError("I will implement this later!")
        h5 = h5py.File(str(cfg["h5_cache"]), 'r')
        val_mask = h5[str(cfg["h5_dset"])][:]
        ret = cls.from_json(cfg["json_cache"], val_mask=val_mask, **kwargs) #TODO: read p_func from cfg
        ret.cfg = cfg
        return ret


class TreeInnervationModelCollection(object):
    def __init__(self, mdl_dict):
        self._mdl_dict = mdl_dict

    def __getitem__(self, item):
        return self._mdl_dict[item]

    @classmethod
    def from_config_file(cls, cfg_file=None):
        import os
        from white_matter.utils.paths_in_config import path_local_to_path
        from white_matter.utils.data_from_config import read_config
        from white_matter.wm_recipe.parcellation import RegionMapper

        if cfg_file is None:
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
            mpr = RegionMapper()
        else:
            mpr = RegionMapper(cfg_file=cfg_file)

        cfg = read_config(cfg_file)
        cfg_root = cfg["cfg_root"]
        cfg = cfg["PTypes"]
        mdl_dict = {}
        for k in cfg.keys():
            path_local_to_path(cfg[k], cfg_root, ["json_cache", "h5_cache"])
            mdl_dict[k] = TreeInnervationModel.from_config(cfg[k], mpr=mpr)
        return cls(mdl_dict)


# VALIDATION OF TREE MODEL
def _naive_model(val_data, smpls=1000):
    N = val_data.shape[1]
    mn_data = val_data.mean(axis=0)
    return numpy.vstack([numpy.random.rand(N) <= mn_data
                         for _ in range(smpls)])


def _make_bins(v):
    if len(numpy.unique(v)) < 1000:
        bins = numpy.unique(v)
    else:
        bins = numpy.linspace(numpy.min(v), numpy.max(v), 999)
    db = numpy.mean(numpy.diff(bins))
    epsilon = db * 1E-9
    bins = numpy.hstack([bins[0] - db, bins, bins[-1] + db])
    return bins, bins[:-1] + db / 2 - epsilon


def distance_func(V, dist='cityblock'):
    from scipy.spatial import distance
    return distance.pdist(V, dist)


def plot_hamming_distances(D_data, D_model, D_naive):
    from matplotlib import pyplot as plt
    bins, bin_c = _make_bins(numpy.hstack([D_data, D_model, D_naive]))
    H_data = numpy.histogram(D_data, bins=bins, density=True)[0]
    H_model = numpy.histogram(D_model, bins=bins, density=True)[0]
    H_naive = numpy.histogram(D_naive, bins=bins, density=True)[0]
    ax = plt.figure().add_axes([0.15, 0.15, 0.8, 0.8])
    ax.plot(bin_c, H_data, label='Data')
    ax.plot(bin_c, H_model, label='Tree-based model')
    ax.plot(bin_c, H_naive, label='Naive model')
    ax.set_xlabel('Hamming distance')
    ax.set_ylabel('Fraction')
    plt.legend()


def to_cdf(v):
    from scipy import interpolate
    bins, bin_c = _make_bins(v)
    H = numpy.histogram(v, bins=bins, density=True)[0]
    H = numpy.cumsum(H) / H.sum()
    return interpolate.interp1d(bin_c, H, 'nearest',
                                bounds_error=False, fill_value='extrapolate')


def to_rvs(smpls):
    def rvs(**kwargs):
        if 'size' not in kwargs:
            return numpy.random.choice(smpls)
        return numpy.random.choice(smpls, kwargs['size'], replace=True)
    return rvs


def validate_tree_model(tree_mdl, val_idx, val_data, smpls=10000, dist='cityblock'):
    from scipy.stats import kstest
    N = val_data.shape[1]

    def idx2bc(idx):
        ret = numpy.zeros(N, dtype=bool)
        ret[idx] = True
        return ret
    grown = [tree_mdl.grow_from(val_idx)
             for _ in range(smpls)]
    grown = numpy.vstack([idx2bc(_x) for _x in grown])
    D_data = distance_func(val_data, dist=dist)
    D_model = distance_func(grown, dist=dist)
    D_naive = distance_func(_naive_model(val_data), dist=dist)
    plot_hamming_distances(D_data, D_model, D_naive)
    # Distances are strongly non-independent samples. Need to use the ORIGINAL number of samples for the "N" kwarg.
    return kstest(to_rvs(D_data), to_cdf(D_model), N=val_data.shape[0]),\
           kstest(to_rvs(D_data), to_cdf(D_naive), N=val_data.shape[0])



