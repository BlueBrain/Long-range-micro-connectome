import requests
import os
from scipy import sparse
import numpy
import pandas


class ConnectomeInstance(object):
    """A class representing a whole neocortex connectome instance with a method to download and instantiate
    a connection matrix representing incoming connections into a specified region."""

    url_pattern = 'https://bbp.epfl.ch/public/mouse-connectome/Instance%d/%s-hemisphere/%s_ALL_INPUTS_%s.%s'
    url_prefix = 'https://bbp.epfl.ch/public/mouse-connectome/'
    suffix = {'matrix': 'csc.npz', 'indices': 'indices.npy'}
    url_neurons = 'https://bbp.epfl.ch/public/mouse-connectome/NEURONS.feather'
    read_methods = {'matrix': sparse.load_npz, 'indices': numpy.load,
                    'neurons': pandas.read_feather}

    def __init__(self, instance, cache_dir=None):
        """Usage:
        I = ConnectomeInstance(1) instantiates the first generated whole-neocortex instance.
        I = ConnectomeInstance(1, cache_dir='path/to/cache' specifies where the downloaded connection
        matrices should be kept."""
        self.instance = instance
        if cache_dir is None:
            self.cache_dir = os.getcwd()

    def _arg2url(self, target_hemisphere, region, source, data_type):
        url_remote = self.url_pattern % (self.instance, target_hemisphere,
                                         region, source, self.suffix[data_type])
        url_local = os.path.join(self.cache_dir, url_remote[len(self.url_prefix):])
        return url_remote, url_local

    def _request(self, fn_remote, fn_local, chunk_size=1024):
        import progressbar
        r = requests.get(fn_remote, stream=True)
        dir_local = os.path.split(fn_local)[0]
        if not os.path.exists(dir_local):
            os.makedirs(dir_local)
        with open(fn_local, 'wb') as f_out:
            data_size = int(r.headers.get('content-length'))
            pbar = progressbar.ProgressBar(maxval=(data_size / chunk_size) + 1,
                                           widgets=[progressbar.FormatLabel(fn_remote[len(self.url_prefix):] + "  "),
                                                    progressbar.SimpleProgress(),
                                                    progressbar.FormatLabel("  "),
                                                    progressbar.ETA()])
            pbar.start()
            i = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f_out.write(chunk)
                    f_out.flush()
                    i += 1
                    pbar.update(i)
            f_out.close()
            r.close()
            pbar.finish()

    def _load_file(self, fn_remote, fn_local, read_method, overwrite=False):
        if not os.path.isfile(fn_local) or overwrite:
            self._request(fn_remote, fn_local)
        return read_method(fn_local)

    def _load(self, target_hemisphere, region, source, data_type, overwrite=False):
        fn_remote, fn_local = self._arg2url(target_hemisphere, region, source, data_type)
        assert data_type in self.read_methods, "data_type must be one of: %s" % str(self.read_methods.keys())
        return self._load_file(fn_remote, fn_local, self.read_methods[data_type],
                               overwrite=overwrite)

    def incoming_connectivity(self, target_hemisphere, region, sources, overwrite=False):
        """Returns a sparse connection matrix of incoming connections into the specified region
        and the indices of the neurons in that region in the whole-neocortex model.
        The matrix has a shape of N x A, where N is the number of neurons is the whole-neocortex
        model and A the number of neurons in the specified region.
        The indices is a numpy.array of length A.

        M, indices = I.incoming_connectivity(target_hemisphere, region, sources),
        where target_hemisphere is one of 'left', or 'right';
        region is one of the brain regions of the Allen Mouse Brain Atlas parcellation scheme.
        sources is a list containing any or all of the following: 'local' for local connectivity
        withing the region, 'ipsi' for long-range connectivity from the ipsi-lateral hemisphere,
        'contra' for long-range connectivity from the contra-lateral hemisphere.

        Example:
            M, indices = I.incoming_connectivity('right', 'ACAd', ['local', 'ipsi', 'contra'])
            for the complete incoming connectivity into the right ACAd.
            """
        indices = None
        M = None
        for src in sources:
            if indices is None:
                indices = self._load(target_hemisphere, region, src, 'indices',
                                     overwrite=overwrite)
            if M is None:
                M = self._load(target_hemisphere, region, src, 'matrix',
                               overwrite=overwrite)
            else:
                M = M + self._load(target_hemisphere, region, src, 'matrix',
                                   overwrite=overwrite)
        return M, indices

