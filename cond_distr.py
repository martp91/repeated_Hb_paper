import numpy as np
from numba import njit, vectorize
import ctypes
from scipy.integrate import nquad, quad
import scipy.stats as scistats

# from scipy.special import erf
from numba import cfunc, types
from scipy import LowLevelCallable
from math import erf

# for lowlevelcallable
c_sig = types.double(types.intc, types.CPointer(types.double))


def correct_std_dev_cond(sig_diff, sig_pop):
    if sig_diff <= 0.2:
        return sig_diff
    return 0.2 + (sig_diff - 0.2) * (1.46 - 0.4 * sig_pop)


SQRT2 = 2**0.5
PREF = 1 / np.sqrt(2 * np.pi)


@vectorize
def measure_hb(x, std_meas, cut, p_repeat_1=1.0, p_repeat_2=1.0, p_sjoemel=0.0):
    """measurement procedure assumed"""
    nd = 1
    x1 = round(np.random.normal(x, std_meas), nd)
    if (x1 >= cut) | (np.random.random() > p_repeat_1):
        return x1
    x2 = round(np.random.normal(x, std_meas), nd)
    # return x2
    if x2 >= cut:
        return x2
    if np.random.random() > p_repeat_2:
        if p_sjoemel > np.random.random():
            return cut
        return max([x1, x2])
    x3 = round(np.random.normal(x, std_meas), nd)
    if x3 >= cut:
        return x3

    if p_sjoemel > np.random.random():
        return cut

    return max([x1, x2, x3])


@njit
def norm_pdf(x, mu, sig):
    return PREF / sig * np.exp(-0.5 * (x - mu) ** 2 / sig**2)


@njit
def log_norm_pdf(x, mu, sig):
    return -0.5 * (x - mu) ** 2 / sig**2 - np.log(sig / PREF)


@vectorize
def norm_cdf(x, mu, sig):
    return 0.5 * (1 + erf((x - mu) / (SQRT2 * sig)))


@njit
def log_norm_cdf_lower(x):
    # very crude approx of log_norm for small x
    # should actually contain another series expansion
    return -0.5 * x**2 - np.log(-x) - 0.5 * np.log(2 * np.pi)


@vectorize
def log_norm_cdf(x, mu, sig):
    # these are the bounds found by experimenting
    lower_seg = -8
    # upper_seg = 10 #upper_seg returns zero anyway
    x_ = (x - mu) / sig
    if x_ > lower_seg:
        return np.log(norm_cdf(x, mu, sig))
    else:
        return log_norm_cdf_lower(x_)


# @njit
@vectorize
def pdf_cond_2nd_meas(x, mu, sig, rho, cut):
    "Prob of xB given that xA < c"
    # Normalization fixed. Should divide by the chance of P(xA < c)
    # Given by mathematica
    # For rho == 0 this simplifies to just a normal distribution
    xp = (x - mu) / sig
    cp = (cut - mu) / sig
    return (
        PREF
        / sig
        * np.exp(-0.5 * xp**2)
        * (1 + erf((cp - rho * xp) / (np.sqrt(2 - 2 * rho**2))))
        / (1 + erf(cp / np.sqrt(2)))
    )


PREF_SQ = 1 / (2 * np.pi)


@njit
def bivariate_normal(x, y, mux, muy, sigx, sigy, rho):
    pref = PREF_SQ * 1 / (sigx * sigy * np.sqrt(1 - rho**2))
    xp = (x - mux) / sigx
    yp = (y - muy) / sigy
    inexp = -0.5 / (1 - rho**2) * (xp**2 + yp**2 - 2 * rho * xp * yp)
    return pref * np.exp(inexp)


@cfunc(c_sig)
def bivariate_normal_cfunc(n, args):
    return bivariate_normal(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    )


bivariate_normal_cfunc = LowLevelCallable(bivariate_normal_cfunc.ctypes)


def ev_x_func(mu, sig, cut):
    "expectation value of x1 when repeating below cut"
    return mu - np.sqrt(2 / np.pi) * sig * np.exp(-0.5 * (cut - mu) ** 2 / sig**2) / (
        1 + erf((cut - mu) / (sig * np.sqrt(2)))
    )


def ev_y_func(mu, sig, rho, cut):
    "expectation value of x2 when repeating x1 below cut"
    return mu - np.sqrt(2 / np.pi) * sig * rho * np.exp(
        -0.5 * (cut - mu) ** 2 / sig**2
    ) / (1 + erf((cut - mu) / (sig * np.sqrt(2))))


@njit
def trivariate_normal(x, y, z, mu, sig, rho):
    # assume 3 measurements from same population
    # so same mu, sig, rho
    w = 1 / sig
    xp = (x - mu) * w
    yp = (y - mu) * w
    zp = (z - mu) * w
    xp2 = xp**2
    yp2 = yp**2
    zp2 = zp**2
    inexp = (
        -xp2 * (1 + rho)
        - yp**2 * (1 + rho)
        - zp**2 * (1 + rho)
        + 2 * yp * zp * rho
        + 2 * xp * (yp + zp) * rho
    ) / ((rho - 1) * (1 + 2 * rho))

    pref = (
        1 / (2 * 2**0.5 * np.pi**1.5) * w**3 * (1 / ((1 - rho) * (1 + 2 * rho) ** 0.5))
    )
    return pref * np.exp(-0.5 * inexp)


@cfunc(c_sig)
def trivariate_normal_cfunc(n, args):
    x = args[0]
    y = args[1]
    z = args[2]
    mu = args[3]
    sig = args[4]
    rho = args[5]
    return trivariate_normal(x, y, z, mu, sig, rho)


trivariate_normal_cfunc_llc = LowLevelCallable(trivariate_normal_cfunc.ctypes)


def norm_cond_trinormal(mu, sig, rho, cut):
    "Normalization of x3 given x2<c and x1 <c"
    # Deprecated use scistats.mvnorm
    return nquad(
        bivariate_normal_cfunc,
        [[-np.inf, cut], [-np.inf, cut]],
        args=(mu, mu, sig, sig, rho, cut),
    )[0]


# same but SLOWER
# def cond_3rd_pdf_2(x, mu, sig, rho, cut, dx=0.05):
#     x_ = np.arange(x.min(), x.max(), dx)
#     cut_ = np.ones_like(x_)*cut
#     var = sig**2
#     varrho = var*rho
#     data = np.dstack([x_, cut_, cut_])
#     F = scistats.multivariate_normal.cdf(data, [mu, mu, mu],
#         [[var, varrho, varrho], [varrho, var, varrho], [varrho, varrho, var]]
#     )
#     f = np.diff(F, prepend=F[0]/2)/dx
#     f /= scistats.multivariate_normal.cdf([cut, cut], [mu, mu], [[var, varrho], [varrho, var]])

#     return np.interp(x, x_, f)


def pdf_cond_3rd_meas(z, mu, sig, rho, cut, norm=None):
    # norm = norm_cond_trinormal(mu, sig, rho, cut)
    if norm is None:
        norm = scistats.multivariate_normal.cdf(
            [cut, cut], [mu, mu], [[sig**2, rho * sig**2], [rho * sig**2, sig**2]]
        )  # slightly faster
    # vectorize if given
    try:
        return (
            np.array(
                [
                    nquad(
                        trivariate_normal_cfunc_llc,
                        [[-np.inf, cut], [-np.inf, cut]],
                        args=(z_, mu, sig, rho),
                        opts=dict(epsabs=1e-5, epsrel=1e-5),  # save some time
                    )[0]
                    for z_ in z
                ]
            )
            / norm
        )
    except TypeError:
        return (
            nquad(
                trivariate_normal_cfunc_llc,
                [[-np.inf, cut], [-np.inf, cut]],
                args=(z, mu, sig, rho, cut),
            )[0]
            / norm
        )


def F3_max(x, mu, sig_tot, rho):
    # cdf of max of 3 correlated normals is same as cdf of triv norm
    var = sig_tot**2
    varrho = var * rho
    # slower
    # try:
    #     return np.array([nquad(trivariate_normal_cfunc_llc, [[-np.inf, x_], [-np.inf, x_], [-np.inf, x_]], args=(mu, sig, rho),
    #                         opts=dict(epsabs=1e-5, epsrel=1e-5))[0] for x_ in x])
    # except TypeError:
    #     return nquad(trivariate_normal_cfunc_llc, [[-np.inf, x], [-np.inf, x], [-np.inf, x]], args=(mu, sig, rho),
    #                                          opts=dict(epsabs=1e-5, epsrel=1e-5))[0]
    return scistats.multivariate_normal.cdf(
        np.dstack([x, x, x]),
        [mu, mu, mu],
        [[var, varrho, varrho], [varrho, var, varrho], [varrho, varrho, var]],
    )


def f3_max(x, mu, sig_tot, rho, dx=0.05):
    # take derivative (but poorly!) of cdf F3
    x_ = np.arange(x.min(), x.max(), dx)
    F3 = F3_max(x_, mu, sig_tot, rho)
    f3 = np.diff(F3, prepend=F3[0] / 2) / dx
    return np.interp(x, x_, f3)


def f2_max(x, mu, sig_tot, rho):
    x_ = (x - mu) / sig_tot
    return 2 * norm_pdf(x_) * norm_cdf((1 - rho) / (1 - rho**2) ** 0.5 * x_)


def calc_p1_below_cut(cut, mu, sig):
    return norm_cdf(cut, mu, sig)


def calc_p2_below_cut(cut, mu, sig, rho):
    var = sig**2
    varrho = var * rho
    return scistats.multivariate_normal.cdf(
        [cut, cut], [mu, mu], [[var, varrho], [varrho, var]]
    )


def calc_p3_below_cut(cut, mu, sig, rho):
    var = sig**2
    varrho = var * rho
    return scistats.multivariate_normal.cdf(
        [cut, cut, cut],
        [mu, mu, mu],
        [[var, varrho, varrho], [varrho, var, varrho], [varrho, varrho, var]],
    )


def pdf_comb(
    x, mu, sig, rho, cut, p_sjoemel=0.0, p_repeat_1=1, p_repeat_2=1, p_repeat_high=0
):
    dr = 0.05  # for rounding
    p1 = calc_p1_below_cut(cut - dr, mu, sig)
    p2 = calc_p2_below_cut(cut - dr, mu, sig, rho) / p1
    p3 = calc_p3_below_cut(cut - dr, mu, sig, rho) / (p1 * p2)

    f1 = norm_pdf(x, mu, sig)
    f2 = pdf_cond_2nd_meas(x, mu, sig, rho, cut - dr)

    # no need to calculate below/above cut
    f3_copy = np.zeros_like(x)
    mask = x < cut - dr
    f3 = pdf_cond_3rd_meas(x[~mask], mu, sig, rho, cut - dr, norm=p2 * p1)
    f3_copy[~mask] = f3
    f3 = f3_copy

    fmax_3_copy = np.zeros_like(x)
    fmax_3 = f3_max(x[mask], mu, sig, rho)
    norm_max_3 = F3_max(cut - dr, mu, sig, rho)
    fmax_3 /= norm_max_3
    fmax_3_copy[mask] = fmax_3
    fmax_3 = fmax_3_copy
    if p_repeat_high > 0:
        fmax_2 = f2_max(x, mu, sig, rho)  # normalize?
    else:
        fmax_2 = 0

    if isinstance(x, np.ndarray):
        f1[x < cut - dr] *= 1 - p_repeat_1
        f2[x < cut - dr] *= 1 - p_repeat_2
        if p_repeat_high > 0:
            fmax_2[x < cut - dr] *= 1 - p_repeat_1
    else:
        raise NotImplementedError("use arrays")

    f_sjoemel = np.where(
        (x > cut - dr) & (x < cut + dr), 1 / (2 * dr), 0
    )  # sjoemel and round to cutoff
    fp1 = f1
    fp2 = (
        p1 * f2 * p_repeat_1 * (1 - p_repeat_high)
        + p1 * p_repeat_1 * fmax_2 * p_repeat_high
    )
    fp3 = p1 * p2 * f3 * p_repeat_1 * p_repeat_2
    fpmax = p1 * p2 * p3 * p_repeat_1 * p_repeat_2 * fmax_3
    return (fp1 + fp2 + fp3 + fpmax) * (1 - p_sjoemel) + p_sjoemel * f_sjoemel


@njit
def pdf_max(x, mu, sig):
    # max of 3 measurements
    return 3 * norm_pdf(x, mu, sig) * norm_cdf(x, mu, sig) ** 2


@njit
def log_pdf_max(x, mu, sig):
    return np.log(3) + log_norm_pdf(x, mu, sig) + 2 * log_norm_cdf(x, mu, sig)


@vectorize
def pdf_repeated_meas(x, mu, sig, cut):
    """For only 1 donor not distr of donor hbs"""
    p1 = norm_cdf(cut - 0.05, mu, sig)
    f1 = norm_pdf(x, mu, sig)
    if x < cut - 0.05:
        return pdf_max(x, mu, sig)
    return f1 * (1 + p1 + p1**2)


@vectorize
def log_p_repeated_meas(x, mu, sig, cut):
    log_f1 = log_norm_pdf(x, mu, sig)
    p1 = norm_cdf(cut - 0.05, mu, sig)
    if x < cut - 0.05:
        return log_pdf_max(x, mu, sig)
    return log_f1 + np.log(1 + p1 + p1**2)


@vectorize
def prob_below(mu, sig, cut):
    "Prob of a measurement below cutoff given mu and sig, checked integral by mathematica"
    return 0.125 * (1 + erf((cut - mu) / (SQRT2 * sig))) ** 3


# Testing
def test_cond_2nd_meas():
    assert np.isclose(pdf_cond_2nd_meas(8, 8.5, 0.6, 0.6, 7.8), 0.781315569)


def test_cond_3rd_meas():
    assert np.isclose(pdf_cond_3rd_meas(8, 8.5, 0.6, 0.6, 7.8), 0.69398536)


@njit
def neg_log_like_repeated_meas(x, mu, sig, cut):
    return -np.sum(log_p_repeated_meas(x, mu, sig, cut))


def test_pdf_comb():
    assert np.all(
        np.isclose(pdf_comb(np.arange(7, 8, 0.1), 8.5, 0.6, 0.6, 7.8), TEST_VALS_COMB)
    )


def test_neg_log_like_repeated_meas():
    assert np.isclose(
        neg_log_like_repeated_meas(np.arange(7, 8, 0.1), 8.5, 0.4, 7.8),
        129.906987,
    )


TEST_VALS_COMB = (
    [
        0.00104761,
        0.00230788,
        0.00485856,
        0.00977603,
        0.01880448,
        0.03458543,
        0.06083521,
        0.10236481,
        0.45360661,
        0.51887815,
    ],
)

test_cond_2nd_meas()
test_cond_3rd_meas()
# test_pdf_comb()
test_neg_log_like_repeated_meas()
