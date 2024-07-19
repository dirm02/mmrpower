import numpy as np
from scipy.stats import norm, ncx2, chi2, poisson, f
from scipy.linalg import svd, qr, inv, eig, LinAlgError
from scipy.integrate import quad
from scipy.special import gammainc, gammaincc
import warnings

def powfun(Nmult, desired_power, alpha, k, n, sigxobs, sigyobs, alphax, alphay, rhoobs, samplingprop):
    """
    Calculate the power for detecting the moderating effect using the provided parameters.
    
    Parameters
    ----------
    Nmult : int
        Multiplier for sample sizes.
    desired_power : list or array-like
        Desired power value(s). If empty, no desired power calculation is performed.
    alpha : float
        Type I error rate.
    k : int
        Number of subpopulations.
    n : list or array-like
        Sample sizes for each subpopulation.
    sigxobs : list or array-like
        Observed standard deviations for X.
    sigyobs : list or array-like
        Observed standard deviations for Y.
    alphax : list or array-like
        Reliabilities for X.
    alphay : list or array-like
        Reliabilities for Y.
    rhoobs : list or array-like
        Observed correlations between X and Y.
    samplingprop : list or array-like
        Truncation proportions for each subpopulation.

    Returns
    -------
    pow_obs : float
        Observed power.
    chsq_mean : float
        Mean of the chi-square distribution.
    chsq_var : float
        Variance of the chi-square distribution.
    """
    
    if any(sp >= 1 for sp in samplingprop):
        raise ValueError("Elements of samplingprop should be less than 1 to avoid division by zero.")

    # Convert lists to column vectors
    n = np.array(n).reshape(-1, 1) * Nmult
    sigxobs = np.array(sigxobs).reshape(-1, 1)
    sigyobs = np.array(sigyobs).reshape(-1, 1)
    alphax = np.array(alphax).reshape(-1, 1)
    alphay = np.array(alphay).reshape(-1, 1)
    rhoobs = np.array(rhoobs).reshape(-1, 1)
    samplingprop = np.array(samplingprop).reshape(-1, 1)

    z0 = norm.ppf(samplingprop)
    rr = norm.pdf(z0) / (1 - np.array(samplingprop))
    delta = 1 + rr * (z0 - rr)
    wgroupvobs = (1 - np.array(rhoobs) ** 2) * (np.array(sigyobs) ** 2)
    
    # Handle the case where n is 1 to avoid division by zero
    d = np.zeros_like(n, dtype=float)
    mask = n != 1
    d[mask] = (n[mask] + 1) / (delta[mask] * ((n[mask] - 1) ** 2) * np.array(sigxobs)[mask] ** 2)
    
    v = d * wgroupvobs
    betaobs = np.array(rhoobs) * np.array(sigyobs) / np.array(sigxobs)

    C = null_space(np.ones((1, k)))
    D = np.diag(d.flatten())
    V = np.diag(v.flatten())
    
    try:
        Mdi = inv(C.T @ D @ C)
    except LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted. Check the input parameters.")

    Mv = C.T @ V @ C
    Om, U = eig(Mdi @ Mv)
    om = np.diag(Om)
    N = np.sum(n)
    betaobs_outer = np.outer(betaobs, betaobs)

    # Ensure that betaobs_outer is a 2D array and compatible with the rest of the matrices
    if betaobs_outer.ndim == 2 and betaobs_outer.shape[0] == C.shape[0]:
        lam = np.diag(U.T @ C.T @ betaobs @ betaobs.T @ C @ U)
        mv_diag = np.diag(Mv)
        if mv_diag.ndim == 1:
            lam = lam / mv_diag
        else:
            raise ValueError("Mv diagonal must be 1-dimensional.")
    else:
        raise ValueError("betaobs_outer must be a square matrix compatible with C.")
    
    lam = np.concatenate((np.zeros(k), lam))

    crit = ((k - 1) / (N - 2 * k)) * f.ppf(1 - alpha, k - 1, N - 2 * k)
    a = np.concatenate((wgroupvobs * crit, -om))
    a = a.flatten().real
    df = np.concatenate((n.flatten() - 2, np.ones(k - 1)))

    chsq_mean = np.sum(a * (df + lam))
    chsq_var = np.sum(2 * (a ** 2 * (df + 2 * lam)))

    Z = (crit - chsq_mean) / np.sqrt(chsq_var)
    pow_obs, _ = gx2cdf(a, df, lam, 0, 0)
    if desired_power:
        pow_obs = (pow_obs - desired_power) ** 2

    return pow_obs[0], chsq_mean, chsq_var

def gx2cdf(w, k, lambda_, s=0, m=0):
    """
    Generalized chi-square cumulative distribution function (CDF).
    
    Parameters
    ----------
    w : array_like
        Weights.
    k : array_like
        Degrees of freedom.
    lambda_ : array_like
        Non-centrality parameters.
    s : float, optional
        Scale parameter.
    m : float, optional
        Mean parameter.

    Returns
    -------
    p : ndarray
        Computed probabilities.
    p_err : ndarray
        Error in the computed probabilities.
    """
    x = [0]
    if len(np.unique(w)) == 1:
        # No s and only one unique weight
        if np.sign(np.unique(w)) == 1:
            return ncx2.cdf((x - m) / np.unique(w), np.sum(k), np.sum(lambda_))
        else:
            return ncx2.cdf((x - m) / np.unique(w), np.sum(k), np.sum(lambda_), 'upper')
    elif not s:
        if np.all(w > 0):
            return gx2_ruben(x, w, k, lambda_, m)
        else:
            return gx2_imhof(x, w, k, lambda_, s, m)
    else:
        return gx2_imhof(x, w, k, lambda_, s, m)

def gx2_ruben(x, w, k, lambda_, m):
    """
    Generalized chi-square CDF using Ruben's method.
    
    Parameters
    ----------
    x : array_like
        Input array.
    w, k, lambda_ : array_like
        Weights, degrees of freedom, and non-centrality parameters.
    m : float
        Mean parameter.

    Returns
    -------
    p : ndarray
        Computed probabilities.
    p_err : ndarray
        Error in the computed probabilities.
    """
    n_ruben = int(1e2)
    x_flat = x.ravel()

    if np.all(w < 0):
        w = -w
        x_flat = -x_flat
        m = -m
        w_pos = False
    else:
        w_pos = True

    beta = 0.90625 * np.min(w)
    M = np.sum(k)
    n = np.arange(1, n_ruben)

    g = np.sum(k * (1 - beta / w) ** n, axis=1) + beta * n * ((1 - beta / w) ** (n - 1)) * (lambda_ / w)

    a = np.empty(n_ruben)
    a[0] = np.sqrt(np.exp(-np.sum(lambda_)) * beta ** M * np.prod(w ** (-k)))
    if a[0] < np.finfo(float).eps:
        raise OverflowError('Underflow error: some series coefficients are smaller than machine precision.')

    for j in range(n_ruben - 1):
        a[j + 1] = np.dot(np.flip(g[:j + 1]), a[:j + 1]) / (2 * (j + 1))

    x_grid, k_grid = np.meshgrid((x_flat - m) / beta, M + 2 * np.arange(n_ruben - 1))

    F = np.vectorize(lambda x, k: chi2.cdf(x, k) if w_pos else chi2.cdf(x, k, 'upper'))(x_grid, k_grid)

    p = np.dot(a, F)

    if w_pos:
        p = p
    else:
        p = p / beta

    p_err = (1 - np.sum(a)) * chi2.cdf((x_flat - m) / beta, M + 2 * n_ruben)

    return p.reshape(x.shape), p_err.reshape(x.shape)

def gx2_imhof_integrand(u, x, w, k, lambda_, s, m, output):
    """
    Define the Imhof integrand.

    Parameters
    ----------
    u : float
        Integration variable.
    x : float
        Input value.
    w, k, lambda_ : array_like
        Column vectors for weights, degrees of freedom, and non-centrality parameters.
    s : float
        Scale parameter.
    m : float
        Mean parameter.
    output : str
        Output type, either 'cdf' or 'pdf'.

    Returns
    -------
    f : float
        Value of the integrand.
    """
    theta = np.sum(k * np.arctan(w * u) + (lambda_ * (w * u)) / (1 + w**2 * u**2), axis=0) / 2 + u * (m - x) / 2
    rho = np.prod(((1 + w**2 * u**2)**(k / 4)) * np.exp(((w**2 * u**2) * lambda_) / (2 * (1 + w**2 * u**2))), axis=0) * np.exp(u**2 * s**2 / 8)
    if output == 'cdf':
        return np.sin(theta) / (u * rho)
    elif output == 'pdf':
        return np.cos(theta) / rho

def gx2_imhof(x, w, k, lambda_, s, m, output='cdf', side='lower', AbsTol=1.0000e-10, RelTol=0.0100):
    """
    Compute the integral using the Imhof method.

    Parameters
    ----------
    x : array_like
        Input array.
    w, k, lambda_ : array_like
        Weights, degrees of freedom, and non-centrality parameters.
    s : float
        Scale parameter.
    m : float
        Mean parameter.
    output : str, optional
        Output type, either 'cdf' or 'pdf'. Default is 'cdf'.
    side : str, optional
        Side of the distribution, either 'lower' or 'upper'. Default is 'lower'.
    AbsTol : float, optional
        Absolute tolerance for the integral. Default is 1.0000e-10.
    RelTol : float, optional
        Relative tolerance for the integral. Default is 0.0100.

    Returns
    -------
    p : ndarray
        Computed probabilities.
    errflag : ndarray
        Error flags indicating if the probability is outside the range [0, 1].
    """
    imhof_integral = np.array([quad(lambda u: gx2_imhof_integrand(u, xi, w.T, k.T, lambda_.T, s, m, output), 0, np.inf, epsabs=AbsTol, epsrel=RelTol)[0] for xi in x])

    if side == 'lower':
        p = 0.5 - imhof_integral / np.pi
    elif side == 'upper':
        p = 0.5 + imhof_integral / np.pi
    else:
        raise ValueError("Invalid value for 'side'. Must be 'lower' or 'upper'.")

    errflag = (p < 0) | (p > 1)
    p = np.clip(p, 0, 1)

    if np.any(errflag):
        p = np.clip(p, 0, 1)
        warnings.warn('Imhof method output(s) too close to limit to compute exactly, so clipping. Check the flag output, and try stricter tolerances.')

    return p, errflag

def ncx2cdf(x, v, delta, uflag=None):
    """
    Non-central chi-square cumulative distribution function (CDF).
    
    Parameters
    ----------
    x : array_like
        Values at which to evaluate the CDF.
    v : array_like
        Degrees of freedom.
    delta : array_like
        Non-centrality parameters.
    uflag : str, optional
        If 'upper', compute the upper tail probability.
        
    Returns
    -------
    p : ndarray
        Computed probabilities.
    """
    if uflag is not None:
        uflag = str(uflag)

    if len(x) != len(v) or len(x) != len(delta):
        raise ValueError("Input size mismatch")

    flag = uflag == 'upper'
    
    # Initialize P to zero
    p = np.zeros_like(x, dtype=np.float64)
    eps1 = np.finfo(np.float32).eps
    rmin = np.finfo(np.float32).tiny
    
    k0 = np.isnan(x) | np.isnan(v) | np.isnan(delta)
    p[k0] = np.nan
    if flag:
        p[(x == np.inf) & ~k0] = 0
        p[(x <= 0) & ~k0] = 1
    else:
        p[(x == np.inf) & ~k0] = 1
    
    p[delta < 0] = np.nan  # can't have negative non-centrality parameter
    p[v < 0] = np.nan      # can't have negative degrees of freedom

    # 0 degrees of freedom at x=0
    k = (v == 0) & (x == 0) & (delta >= 0) & ~k0
    if flag:
        p[k] = -np.expm1(-delta[k] / 2)
    else:
        p[k] = np.exp(-delta[k] / 2)
    
    # Central chi2cdf
    k = (v >= 0) & (x > 0) & (delta == 0) & np.isfinite(x) & ~k0
    if flag:
        p[k] = gammaincc(v[k] / 2, x[k] / 2)
    else:
        p[k] = gammainc(v[k] / 2, x[k] / 2)
    
    # Normal case
    todo = np.where((v >= 0) & (x > 0) & (delta > 0) & np.isfinite(x) & ~k0)[0]
    delta = delta[todo] / 2
    v = v[todo] / 2
    x = x[todo] / 2

    # Compute Chernoff bounds
    e0 = np.log(rmin)
    e1 = np.log(eps1 / 4)
    t = 1 - (v + np.sqrt(v**2 + 4 * delta * x)) / (2 * x)
    q = delta * t / (1 - t) - v * np.log(1 - t) - t * x
    peq0 = (x < delta + v) & (q < e0)
    peq1 = (x > delta + v) & (q < e1)

    if flag:
        p[todo[peq0]] = 1
    else:
        p[todo[peq1]] = 1

    todo = todo[~(peq0 | peq1)]
    x = x[~(peq0 | peq1)]
    v = v[~(peq0 | peq1)]
    delta = delta[~(peq0 | peq1)]

    # Find index K of the maximal term in the summation series
    K1 = np.ceil((np.sqrt((v + x)**2 + 4 * x * delta) - (v + x)) / 2)
    K = np.zeros_like(x, dtype=int)
    k1above1 = np.where(K1 > 1)[0]
    K2 = np.floor(delta[k1above1] * gammaincratio(x[k1above1], K1[k1above1]))
    fixK2 = np.isnan(K2) | np.isinf(K2)
    K2[fixK2] = K1[k1above1[fixK2]]
    K[k1above1] = K2.astype(int)

    if flag:
        k0 = (K == 0) & (v == 0)
        K[k0] = 1

    pois = poisson.pmf(K, delta)

    if flag:
        # Compute upper tail
        full = pois * gammaincc(v + K, x)
    else:
        full = pois * gammainc(v + K, x)

    # Sum the series
    sumK = np.zeros_like(x)
    poisterm = pois
    fullterm = full
    keep = (K > 0) & (fullterm > 0)
    k = K

    while np.any(keep):
        poisterm[keep] = poisterm[keep] * k[keep] / delta[keep]
        k[keep] = k[keep] - 1
        if flag:
            fullterm[keep] = poisterm[keep] * gammaincc(v[keep] + k[keep], x[keep])
        else:
            fullterm[keep] = poisterm[keep] * gammainc(v[keep] + k[keep], x[keep])
        sumK[keep] = sumK[keep] + fullterm[keep]
        keep = keep & (k > 0) & (fullterm > np.finfo(float).eps * sumK)

    poisterm = pois
    fullterm = full
    keep = fullterm > 0
    k = K

    while np.any(keep):
        k[keep] = k[keep] + 1
        poisterm[keep] = poisterm[keep] * delta[keep] / k[keep]
        if flag:
            fullterm[keep] = poisterm[keep] * gammaincc(v[keep] + k[keep], x[keep])
        else:
            fullterm[keep] = poisterm[keep] * gammainc(v[keep] + k[keep], x[keep])
        sumK[keep] = sumK[keep] + fullterm[keep]
        keep = keep & (fullterm > np.finfo(float).eps * sumK)

    p[todo] = full + sumK
    p[p > 1] = 1

    return p

def gammaincratio(x, K1):
    """
    Ratio of incomplete gamma function values at S and S-1.
    
    Parameters
    ----------
    x : array_like
        Input values.
    K1 : array_like
        Degrees of freedom.

    Returns
    -------
    r : ndarray
        Ratio of gammainc values.
    """
    return gammaincc(K1, x) / gammainc(K1, x)

def null_space(A, tol=None, method='orthonormal'):
    """
    Null space of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix.
    tol : float, optional
        Tolerance for singular values to consider as zero. Default is max(size(A)) * eps(norm(A)).
    method : str, optional
        Method to compute the null space: 'orthonormal' (default) or 'rational'.

    Returns
    -------
    Z : ndarray
        Orthonormal or rational basis for the null space of A.
    """
    m, n = A.shape
    use_rational = False

    if method == 'rational':
        use_rational = True

    if use_rational:
        # Rational basis
        Q, R, pivcol = qr(A, pivoting=True, mode='economic')
        rank = np.sum(np.abs(np.diag(R)) > np.finfo(R.dtype).eps * 100 * max(R.shape))
        pivcol = pivcol[:rank]
        nopiv = np.setdiff1d(np.arange(n), pivcol)
        Z = np.zeros((n, n - rank))
        if n > rank:
            Z[nopiv, :] = np.eye(n - rank)
            if rank > 0:
                Z[pivcol, :] = -R[:rank, nopiv]
    else:
        # Orthonormal basis
        U, s, Vt = svd(A)
        if tol is None:
            tol = max(m, n) * np.finfo(float).eps * np.linalg.norm(s, np.inf)
        rank = np.sum(s > tol)
        Z = Vt.T[:, rank:]

    return Z

# Example usage:
if __name__ == '__main__':
    Nmult = 1
    desired_power = []
    alpha = 0.05
    k = 2
    n = [5, 5]
    sigxobs = [0.1, 0.1]
    sigyobs = [0.1, 0.1]
    alphax = [0.1, 0.1]
    alphay = [0.1, 0.1]
    rhoobs = [0.1, 0.1]
    TruncProp = [0.4, 0.4]

    power = powfun(Nmult, desired_power, alpha, k, n, sigxobs, sigyobs, alphax, alphay, rhoobs, TruncProp)
    print("Power:", power)
