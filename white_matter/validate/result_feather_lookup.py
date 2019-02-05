import pandas
import numpy
import bluepy
from white_matter.wm_recipe import region_mapper


class ProjectionizerResult(object):

    def __init__(self, path_feather, path_circ):
        self._data = pandas.read_feather(path_feather, columns=['sgid', 'tgid', 'x', 'y', 'z'])
        self._circ = bluepy.Circuit(path_circ)
        self._mpr = region_mapper.RegionMapper()
        self._post_gids = self._data['tgid'].unique()
        self._pre_gids = self._data['sgid'].unique()

    @staticmethod
    def _combine_dicts(lst_dicts):
        if len(lst_dicts) == 0:
            return {}
        for _d in lst_dicts[1:]:
            for k in _d.keys():
                lst_dicts[0].setdefault(k, []).extend(_d.pop(k))
        return lst_dicts[0]

    def _truncate_layer(self, lst_region_names):
        def _sngl(s):
            contained = [_x in s for _x in self._mpr.region_names]
            if numpy.any(contained):
                return self._mpr.region_names[numpy.nonzero(contained)[0][0]]
            return s
        return numpy.array(map(_sngl, lst_region_names))

    def _presynaptic_property(self, post_gid, property_str, unique=False):
        if hasattr(post_gid, '__iter__'):
            prop = self._data[numpy.in1d(self._data['tgid'], post_gid)][property_str]
        else:
            prop = self._data[self._data['tgid'] == post_gid][property_str]
        if unique:
            return prop.unique()
        return prop

    def _presynaptic_circ_property(self, post_gid, property_str,
                                   unique_neurons=False, unique_properties=False):
        sgids = self._presynaptic_property(post_gid, 'sgid', unique=unique_neurons)
        res = self._circ.v2.cells.get(group=sgids, properties=property_str)
        if not unique_neurons:
            res = res.loc[sgids]
        if unique_properties:
            return res.unique()
        return res

    def _postsynaptic_property(self, pre_gid, property_str, unique=False):
        if hasattr(pre_gid, '__iter__'):
            prop = self._data[numpy.in1d(self._data['sgid'], pre_gid)][property_str]
        else:
            prop = self._data[self._data['sgid'] == pre_gid][property_str]
        if unique:
            return prop.unique()
        return prop

    def _postsynaptic_circ_property(self, pre_gid, property_str,
                                   unique_neurons=False, unique_properties=False):
        tgids = self._postsynaptic_property(pre_gid, 'tgid', unique=unique_neurons)
        res = self._circ.v2.cells.get(group=tgids, properties=property_str)
        if not unique_neurons:
            res = res.loc[tgids]
        if unique_properties:
            return res.unique()
        return res

    def presynaptic_gids(self, post_gids, split=True, **kwargs):
        if hasattr(post_gids, '__iter__') and split:
            return [self._presynaptic_property(_g, 'sgid', **kwargs)
                    for _g in post_gids]
        return self._presynaptic_property(post_gids, 'sgid', **kwargs)

    def presynaptic_syns_con(self, post_gids, split=True, lookup_by='region'):
        if hasattr(post_gids, '__iter__') and split:
            return [self.presynaptic_syns_con(_g, lookup_by=lookup_by)
                    for _g in post_gids]
        if hasattr(post_gids, '__iter__'):
            props = self._presynaptic_property(post_gids, ['sgid', 'tgid'], unique=False)
            idxx = props['sgid'].astype(numpy.int64) + props['tgid'].astype(numpy.int64)\
                   * (props['sgid'].max() + 1)
        else:
            idxx = self.presynaptic_gids(post_gids, unique=False)
        if lookup_by == 'brain region':
            splt = self._presynaptic_circ_property(post_gids, 'region', unique_neurons=False)
            splt = self._truncate_layer(splt)
        else:
            splt = self._presynaptic_circ_property(post_gids, lookup_by, unique_neurons=False).values
        u_splt = numpy.unique(splt)
        return dict([(_reg, idxx[splt == _reg].value_counts().values)
                     for _reg in u_splt])

    def postsynaptic_gids(self, pre_gids, split=True, **kwargs):
        if hasattr(pre_gids, '__iter__') and split:
            return [self._postsynaptic_property(_g, 'tgid', **kwargs)
                    for _g in pre_gids]
        return self._postsynaptic_property(pre_gids, 'sgid', **kwargs)

    def postsynaptic_syns_con(self, pre_gids, split=True, lookup_by='region'):
        if hasattr(pre_gids, '__iter__') and split:
            return [self.postsynaptic_syns_con(_g, lookup_by=lookup_by)
                    for _g in pre_gids]
        if hasattr(pre_gids, '__iter__'):
            props = self._postsynaptic_property(pre_gids, ['sgid', 'tgid'], unique=False)
            idxx = props['sgid'].astype(numpy.int64) + props['tgid'].astype(numpy.int64)\
                   * (props['sgid'].max() + 1)
        else:
            idxx = self.postsynaptic_gids(pre_gids, unique=False)
        if lookup_by == 'brain region':
            splt = self._postsynaptic_circ_property(pre_gids, 'region', unique_neurons=False)
            splt = self._truncate_layer(splt)
        else:
            splt = self._postsynaptic_circ_property(pre_gids, lookup_by, unique_neurons=False).values
        u_splt = numpy.unique(splt)
        return dict([(_reg, idxx[splt == _reg].value_counts().values)
                     for _reg in u_splt])

    def presynaptic_locations(self, post_gids, split=True, **kwargs):
        if hasattr(post_gids, '__iter__') and split:
            return [self._presynaptic_property(_g, ['x', 'y', 'z'], **kwargs)
                    for _g in post_gids]
        return self._presynaptic_property(post_gids, ['x', 'y', 'z'], **kwargs)

    def postsynaptic_locations(self, pre_gids, split=True, **kwargs):
        if hasattr(pre_gids, '__iter__') and split:
            return [self._postsynaptic_property(_g, ['x', 'y', 'z'], **kwargs)
                    for _g in pre_gids]
        return self._postsynaptic_property(pre_gids, ['x', 'y', 'z'], **kwargs)

    def presynaptic_neuron_locations(self, post_gids, split=True, **kwargs):
        if hasattr(post_gids, '__iter__') and split:
            return [self._presynaptic_circ_property(_g, ['x', 'y', 'z'], **kwargs)
                    for _g in post_gids]
        return self._presynaptic_circ_property(post_gids, ['x', 'y', 'z'], **kwargs)

    def postsynaptic_neuron_locations(self, pre_gids, split=True, **kwargs):
        if hasattr(pre_gids, '__iter__') and split:
            return [self._postsynaptic_circ_property(_g, ['x', 'y', 'z'], **kwargs)
                    for _g in pre_gids]
        return self._postsynaptic_circ_property(pre_gids, ['x', 'y', 'z'], **kwargs)

    def to_csc_matrix(self, return_col_indices=True):
        from scipy import sparse
        assert self._data['tgid'].is_monotonic_increasing
        idxx = numpy.hstack([0,
                             numpy.nonzero(numpy.diff(self._data['tgid']) > 0)[0] + 1,
                             len(self._data)])
        shape = (self._circ.v2.cells.count(), len(idxx) - 1)
        M = sparse.csc_matrix((numpy.ones(len(self._data), dtype=bool),
                               self._data['sgid'],
                               idxx), shape)
        if return_col_indices:
            u_tgid = self._data['tgid'].loc[idxx[:-1]].values
            return M, u_tgid
        return M

