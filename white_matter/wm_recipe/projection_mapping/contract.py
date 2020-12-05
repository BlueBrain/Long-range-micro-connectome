import numpy
from .barycentric import BarycentricCoordinates


def contract_min(pts, xy, target_d=0.075):
    from scipy.spatial import distance_matrix
    split_view = numpy.split(pts, numpy.arange(3, len(pts), 3))
    mn = xy.mean(axis=0)

    counter = 0
    bary = BarycentricCoordinates(*split_view).cart2bary(*xy.transpose())
    tgts = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    oldD = numpy.NaN
    D = distance_matrix(tgts, bary).min(axis=0)
    while D.min() > target_d and not oldD < D.min() and counter < 50:
        span = D.max() - D.min()
        fac = span / D.max()
        for i in range(3):
            _of_point(split_view, i, mn, fac)
        bary = BarycentricCoordinates(*split_view).cart2bary(*xy.transpose())
        oldD = D.min()
        D = distance_matrix(tgts, bary).min(axis=0)
        counter += 1
    return split_view


def estimate_mapping_var(data, model):
    import colorsys
    hsl_data = numpy.vstack([colorsys.rgb_to_hls(*_d)
                            for _d in data])
    hsl_model = numpy.vstack([colorsys.rgb_to_hls(*_d)
                             for _d in model])
    ratio_hue_sd = numpy.std(hsl_model[:, 0]) / numpy.std(hsl_data[:, 0])
    ratio_saturation = hsl_model[:, 0].mean() / hsl_data[:, 0].mean()
    return 3 * numpy.sqrt(ratio_hue_sd ** 2 + ratio_saturation ** 2)


def _D(x, y, c):
    return numpy.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)


def _proj_cog(x, y, i, cog):
    tmp = BarycentricCoordinates(x, y)
    idx = numpy.setdiff1d(range(3), i)
    proj = numpy.zeros(3, dtype=float)
    proj[idx] = tmp.cart2bary(*cog)[0, idx]
    proj /= proj.sum()
    return tmp.bary2cart(*proj)[0]


def _around_point(x, y, i, cog, tgt_fac):
    c = _proj_cog(x, y, i, cog)
    idx = numpy.setdiff1d(range(3), i)
    x[idx] = c[0] + tgt_fac * (x[idx] - c[0])
    y[idx] = c[1] + tgt_fac * (y[idx] - c[1])
    return x, y


def _of_point(split_view, i, cog, fac):
    for j, coord in enumerate(split_view):
        coord[i] = cog[j] + fac * (coord[i] - cog[j])
    return split_view


def expand(x_in, y_in, xy):
    cog = xy.mean(axis=0)
    bary = BarycentricCoordinates(x_in, y_in)
    A = bary.cart2bary(xy[:, 0], xy[:, 1])
    req_factor = numpy.percentile(A, 95, axis=0) / 1.33
    if numpy.sum(req_factor > 1) >= 2:
        ijk = numpy.argsort(req_factor)
        fac = numpy.minimum(numpy.mean(req_factor[ijk[1:]]), 2.5)
        x_in, y_in = _around_point(x_in, y_in, ijk[0], cog, fac)
    return x_in, y_in


def contract(x_in, y_in, xy, info_log):
    from scipy.spatial.distance import pdist
    cog = xy.mean(axis=0)
    original = BarycentricCoordinates(x_in, y_in).area()

    dx, dy = numpy.max(xy, axis=0) - numpy.min(xy, axis=0)
    tgt_D = numpy.sqrt(dx ** 2 + dy ** 2)
    final_D = 1.2 * numpy.sqrt(len(xy) / numpy.pi)

    '''info_log.info("\tFirst contraction step")
    D = _D(x_in, y_in, cog)
    fac = 0.75 * tgt_D / D.min()
    if fac < 1.0:
        x_in = cog[0] + fac * (x_in - cog[0])
        y_in = cog[1] + fac * (y_in - cog[1])'''

    nsteps = 0
    maxsteps = 1000
    info_log.info("\tContracting individual sides")
    while nsteps < maxsteps:
        D = _D(x_in, y_in, cog)
        pD = pdist(numpy.vstack([x_in, y_in]).transpose())[-1::-1]
        if numpy.mean(D) < tgt_D:
            break
        i = numpy.argsort(pD)
        tgt_fac = 0.95 * pD[i[-2]] / pD[i[-1]]
        x_in, y_in = _around_point(x_in, y_in, i[-1], cog, tgt_fac)
        nsteps += 1
    info_log.info("\t\tPerformed %d operations\n" % nsteps)
    D = _D(x_in, y_in, cog)
    fac = final_D / D.mean()
    x_in = cog[0] + fac * (x_in - cog[0])
    y_in = cog[1] + fac * (y_in - cog[1])
    x_in, y_in = expand(x_in, y_in, xy)
    final = BarycentricCoordinates(x_in, y_in).area()
    return x_in, y_in, (numpy.sqrt(original / final) - 1) * final_D * 0.5
