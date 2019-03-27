import requests
import os
from scipy import sparse
import numpy
import pandas


class ConnectomeInstance(object):
    """A class representing a whole neocortex connectome instance with a method to download and instantiate
    a connection matrix representing incoming connections into a specified region."""

    url_pattern = 'Instance%d/%s-hemisphere/%s_ALL_INPUTS_%s.%s'
    url_prefix = 'https://bbp.epfl.ch/public/mouse-connectome/'
    url_list_files = 'Instance%d/__files'
    url_list_instances = '__instances'
    suffix = {'matrix': 'csc.npz', 'indices': 'indices.npy'}
    url_neurons = 'https://bbp.epfl.ch/public/mouse-connectome/NEURONS.feather'
    read_methods = {'matrix': sparse.load_npz, 'indices': numpy.load,
                    'neurons': pandas.read_feather}
    valid_values = {'target_hemisphere': ['left', 'right'],
                    'sources': ['ipsi', 'contra', 'local'],
                    'data_type': ['matrix', 'indices']}

    def __init__(self, instance, cache_dir=None):
        """Usage:
        I = ConnectomeInstance(1) instantiates the first generated whole-neocortex instance.
        I = ConnectomeInstance(1, cache_dir='path/to/cache' specifies where the downloaded connection
        matrices should be kept."""
        assert isinstance(instance, int), "Instance must be an integer."
        self.instance = instance
        self._instance_files = []
        if cache_dir is None:
            self.cache_dir = os.getcwd()
        self._read_available_files()

    def _read_available_files(self):
        r = requests.get(self.url_prefix + self.url_list_files % self.instance)
        if r.status_code == 404:
            raise ValueError("The specified connectome instance does not exist! Try one of %s"
                             % str(ConnectomeInstance.available_instances()))
        for ln in r.iter_lines():
            self._instance_files.append(ln.strip())

    def _arg2url(self, target_hemisphere, region, source, data_type):
        url_file = self.url_pattern % (self.instance, target_hemisphere,
                                       region, source, self.suffix[data_type])
        url_remote = self.url_prefix + url_file
        url_local = os.path.join(self.cache_dir, url_file)
        return url_file, url_remote, url_local

    def _request(self, fn, fn_remote, fn_local, chunk_size=1024):
        import progressbar
        r = requests.get(fn_remote, stream=True)
        dir_local = os.path.split(fn_local)[0]
        if not os.path.exists(dir_local):
            os.makedirs(dir_local)
        with open(fn_local, 'wb') as f_out:
            data_size = int(r.headers.get('content-length'))
            pbar = progressbar.ProgressBar(maxval=(data_size / chunk_size) + 1,
                                           widgets=[progressbar.FormatLabel(fn + "  "),
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

    def _load_file(self, fn, fn_remote, fn_local, read_method, overwrite=False):
        if not os.path.isfile(fn_local) or overwrite:
            self._request(fn, fn_remote, fn_local)
        return read_method(fn_local)

    def _load(self, target_hemisphere, region, source, data_type, overwrite=False, force=False):
        self._check_query(target_hemisphere, region, source, data_type)
        fn, fn_remote, fn_local = self._arg2url(target_hemisphere, region, source, data_type)
        assert data_type in self.read_methods, "data_type must be one of: %s" % str(self.read_methods.keys())
        if not force and fn not in self._instance_files:
            raise ValueError("Request could not be completed. Maybe an invalid region specified? Else try force=True.")
        return self._load_file(fn, fn_remote, fn_local, self.read_methods[data_type],
                               overwrite=overwrite)

    def _check_query(self, target_hemisphere, region, source, data_type):
        assert target_hemisphere in self.valid_values['target_hemisphere'],\
        "target_hemisphere must be one of %s" % str(self.valid_values['target_hemisphere'])
        assert source in self.valid_values['sources'], \
        "source must be one of %s" % str(self.valid_values['sources'])
        assert data_type in self.valid_values['data_type'],\
        "data_type must be one of %s" % str(self.valid_values['data_type'])

    def incoming_connectivity(self, target_hemisphere, region, sources, overwrite=False, force=False):
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
        if isinstance(sources, str):
            sources = [sources]
        for src in sources:
            if indices is None:
                indices = self._load(target_hemisphere, region, src, 'indices',
                                     overwrite=overwrite, force=force)
            if M is None:
                M = self._load(target_hemisphere, region, src, 'matrix',
                               overwrite=overwrite, force=force)
            else:
                M = M + self._load(target_hemisphere, region, src, 'matrix',
                                   overwrite=overwrite, force=force)
        return M, indices

    @classmethod
    def available_instances(cls):
        url = cls.url_prefix + cls.url_list_instances
        r = requests.get(url)
        assert r.status_code != 404, "Unexpected error with the remote server"
        instances = map(int, r.iter_lines())
        return instances

