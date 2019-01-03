import os
import mcmodels
import numpy

q_str = "http://connectivity.brain-map.org/projection/csv?criteria=service::mouse_connectivity_target_spatial[injection_structures$eq%d][seed_point$eq%d,%d,%d][primary_structure_only$eq%s]"#[transgenic_lines$eq0]"


class StreamlineDownloader(object):

    def __init__(self, manifest_file=None, tmp_dir=None):
        self.__set_cache__(manifest_file)
        self.__set_tmp_dir__(tmp_dir)

    def __set_cache__(self, manifest_file):
        if manifest_file is None:
            manifest_file = os.path.join(os.getenv("HOME", '.'), 'data/connectivity/voxel_model_manifest.json')
        if not os.path.exists(os.path.split(manifest_file)[0]):
            os.makedirs(os.path.split(manifest_file)[0])
        self._cache = mcmodels.core.VoxelModelCache(manifest_file=manifest_file)
        self._tree = self._cache.get_structure_tree()
        self._vol, self._vol_spec = self._cache.get_annotation_volume()
        self._lr_cutoff = self._vol.shape[-1] / 2

    def __set_tmp_dir__(self, tmp_dir):
        if tmp_dir is None:
            tmp_dir = os.path.join(os.getenv("HOME", '.'), 'data/connectivity/sl_cache')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        self._sl_cache = tmp_dir

    def __region2center__(self, region_acronym, hemisphere='right'):
        idxx = self._tree.get_structures_by_acronym([region_acronym])[0]['id']
        mask = numpy.in1d(self._vol, self._tree.descendant_ids([idxx])[0]).reshape(self._vol.shape)
        if hemisphere == 'right':
            mask[:, :, :self._lr_cutoff] = False
        elif hemisphere == 'left':
            mask[:, :, self._lr_cutoff:] = False
        else:
            raise ReferenceError("Unknown hemisphere: %s" % hemisphere)
        return numpy.vstack(numpy.nonzero(mask)).mean(axis=1)

    def __region2center_coord__(self, region_acronym, hemisphere='right'):
        center = self.__region2center__(region_acronym, hemisphere=hemisphere)
        ret = center * numpy.matrix(self._vol_spec['space directions']).astype(float)
        return numpy.array(ret)[0]

    def __execute_query__(self, q):
        import subprocess, hashlib
        fn = "tmp_sl_" + hashlib.md5(q).hexdigest() + ".csv"
        fn = os.path.join(self._sl_cache, fn)
        if os.path.isfile(fn):
            return fn
        command = ["wget", "-O", fn, q]
        print command
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        retcode = proc.wait()
        if retcode > 0:
            print q
            raise RuntimeError("Error executing query")
        return fn

    def __coords2hemi__(self, pt):
        M = numpy.matrix(self._vol_spec['space directions']).astype(int)
        raw_pt = numpy.diag(pt / M)
        if raw_pt[2] >= self._lr_cutoff:
            return 'right'
        return 'left'

    def add_hemisphere_info(self, a_result):
        """Look up for a dict of streamlines which hemisphere they originate in.
        Returns a dict where the keys are a tuple of the original keys and a string
        specifying the hemisphere"""
        out = {}
        for k, v in a_result.items():
            for sl in v:
                out.setdefault((k, self.__coords2hemi__(sl[-1])), []).append(sl)
        return out

    def get_query_string(self, target_spec, target_hemisphere='right', source_spec='grey', primary_only=True):
        """Build a url to query the Allen servers for a set of streamlines from / to specified regions.
        INPUT:
        target_spec: Specify the approximate endpoint of streamlines. Either a string naming a brain region (VISp, etc.)
        or a list of three coordinates. If a string is specified the center of that region in the specified hemisphere is used.
        target_hemisphere: Only if target_spec is a string.
        source_spec: Either a string naming a brain region (SSp-ll, etc.) or a brain region id.
        primary_only: If true then the primary injection structure must be according to source spec.

        RETURNS:
            a query string """
        if isinstance(source_spec, str):
            source_spec = self._tree.get_structures_by_acronym([source_spec])[0]['id']
        if isinstance(target_spec, str):
            target_spec = self.__region2center_coord__(target_spec, hemisphere=target_hemisphere).astype(int)
        return q_str % ((source_spec, ) + tuple(target_spec) + (str(primary_only).lower(), ))

    def query(self, target_spec, target_hemisphere='right', source_spec='grey',
              primary_only=True, add_hemispheres=True):
        """Query Allen servers for streamlines from / to specified regions.
        INPUT:
        target_spec: Specify the approximate endpoint of streamlines. Either a string naming a brain region (VISp, etc.)
        or a list of three coordinates. If a string is specified the center of that region in the specified hemisphere is used.
        target_hemisphere: Only if target_spec is a string.
        source_spec: Either a string naming a brain region (SSp-ll, etc.) or a brain region id.
        primary_only: If true then the primary injection structure must be according to source spec.
        add_hemispheres: If true then separate sets of streamlines are returned for each hemisphere.

        RETURNS:
            A dict where keys are source regions (or a tuple of source region and source hemisphere) and values the
            streamlines (lists of Nx3 numpy.arrays)
        """
        q = self.get_query_string(target_spec, target_hemisphere=target_hemisphere,
                                  source_spec=source_spec, primary_only=primary_only)
        fn = self.__execute_query__(q)
        sls = self.import_streamlines_from_csv(fn)
        if add_hemispheres:
            return self.add_hemisphere_info(sls)
        return sls

    def query_lengths(self, target_spec, target_hemisphere='right', source_spec='grey',
                      primary_only=True, add_hemispheres=True):
        """Look up lengths of streamlines from / to specified regions.
        INPUT:
        target_spec: Specify the approximate endpoint of streamlines. Either a string naming a brain region (VISp, etc.)
        or a list of three coordinates. If a string is specified the center of that region in the specified hemisphere is used.
        target_hemisphere: Only if target_spec is a string.
        source_spec: Either a string naming a brain region (SSp-ll, etc.) or a brain region id.
        primary_only: If true then the primary injection structure must be according to source spec.
        add_hemispheres: If true then separate sets of streamline lengths are returned for each hemisphere.

        RETURNS:
            A dict where keys are source regions (or a tuple of source region and source hemisphere) and values the
            lengths of streamlines (lists floats)
                """
        res = self.query(target_spec, target_hemisphere=target_hemisphere, source_spec=source_spec,
                         primary_only=primary_only, add_hemispheres=add_hemispheres)
        return dict([(k, map(self.streamline_length, v)) for k, v in res.items()])

    @staticmethod
    def import_streamlines_from_csv(filename):
        import csv
        import json
        streams = {}
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)  # skip the headers
            for row in reader:
                stream = []
                streamline = row[16].replace('=>', ':')
                source_region = row[4]
                j = json.loads(streamline)
                for c in j:
                    point = c['coord']
                    stream.append([(float)(point[0]), (float)(point[1]), (float)(point[2])])
                streams.setdefault(source_region, []).append(numpy.array(stream))
            csvfile.close()
        return streams

    @staticmethod
    def streamline_length(a_line):
        return numpy.sqrt(numpy.sum(numpy.diff(a_line, axis=0) ** 2, axis=1)).sum()
