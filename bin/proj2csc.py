#!/usr/bin/env python


def main(path_in, path_circ, prefix_out=None):
    import os, numpy
    from scipy import sparse
    assert os.path.isfile(path_in) and os.path.isfile(path_circ)
    if prefix_out is None:
        prefix_out = os.path.splitext(os.path.abspath(path_in))[0]
    matrix_out = prefix_out + '.csc.npz'
    indices_out = prefix_out + '.indices.npy'
    assert path_in != matrix_out and path_in != indices_out
    from white_matter.validate import ProjectionizerResult
    R = ProjectionizerResult(path_in, path_circ)
    M, indices = R.to_csc_matrix(return_col_indices=True)
    sparse.save_npz(matrix_out, M)
    numpy.save(indices_out, indices)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print """%s -- converts the full output of the white matter projection generator
        (all synapses and their locations) into a scipy.sparse connection matrix (25 times smaller).
        
        Usage: %s path_to_projections.feather path_to_circuit_config""" % (__file__, __file__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
