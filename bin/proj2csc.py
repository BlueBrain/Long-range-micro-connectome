#!/usr/bin/env python


def main(path_in, path_circ, prefix_out=None):
    import os, numpy
    from scipy import sparse
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
        print "Usage: %s path_to_projections.feather path_to_circuit_config"
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
