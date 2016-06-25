import math
import numpy as np
import theano
import theano.tensor as T


def my_logsumexp(a):
    shape = list(a.shape)
    shape[len(shape)-1] = 1
    a_max = np.max(a, axis=-1)
    a_sum = np.sum(np.exp(a - a_max.reshape(tuple(shape))), axis=-1)
    return a_max + np.log(a_sum)


def my_logaddexp(a, b):
    tmp = a - b
    return np.where(tmp > 0, a + np.log1p(np.exp(-tmp)), b + np.log1p(np.exp(tmp)))


def my_logsumexp_theano(a):
    a_max = T.max(a, axis=-1, keepdims=True)
    a_sum = T.sum(np.exp(a - a_max), axis=-1)
    return T.max(a, axis=-1) + T.log(a_sum)


def my_logaddexp_theano(a, b):
    tmp = a - b
    return T.switch(tmp > 0, a + T.log1p(T.exp(-tmp)), b + T.log1p(T.exp(tmp)))


def update_workspace(workspace, N_R, N_I, N_S, N_T):
    if workspace is None:
        workspace = {'N_R': 0, 'N_I': 0, 'N_S': 0, 'N_T': 0}

    # if N_R is not None and workspace['N_R'] < N_R or workspace['N_T'] != N_T:
    #     workspace['sigma2_R'] = np.empty((N_R, N_T), dtype=np.float64)
    #     workspace['correlation_R'] = np.empty((N_R, N_T), dtype=np.float64)
    #     workspace['power_R'] = np.empty((N_R, N_T), dtype=np.float64)
    #     if workspace['N_R'] < N_R:
    #         workspace['e_R'] = np.empty((N_R,), dtype=np.float64)
    #         workspace['avgphi_R'] = np.empty((N_R,), dtype=np.float64)
    #     workspace['N_R'] = N_R
    #
    # if N_I is not None and (workspace['N_I'] < N_I or workspace['N_T'] != N_T):
    #     workspace['sigma2_I'] = np.empty((N_I, N_T), dtype=np.float64)
    #     workspace['correlation_I'] = np.empty((N_I, N_T), dtype=np.float64)
    #     workspace['power_I'] = np.empty((N_I, N_T), dtype=np.float64)
    #     workspace['g_I'] = np.empty((N_I, N_T), dtype=np.complex64)
    #     if workspace['N_I'] < N_I:
    #         workspace['e_I'] = np.empty((N_I,), dtype=np.float64)
    #         workspace['avgphi_I'] = np.empty((N_I,), dtype=np.float64)
    #     workspace['N_I'] = N_I
    #
    # if N_S is not None and (workspace['N_S'] < N_S or workspace['N_T'] != N_T):
    #     workspace['sigma2_S'] = np.empty((N_S, N_T), dtype=np.float64)
    #     workspace['correlation_S'] = np.empty((N_S, N_T), dtype=np.float64)
    #     workspace['power_S'] = np.empty((N_S, N_T), dtype=np.float64)
    #     workspace['g_S'] = np.empty((N_S, N_T), dtype=np.complex64)
    #     if workspace['N_S'] < N_S:
    #         workspace['e_S'] = np.empty((N_S,), dtype=np.float64)
    #         workspace['avgphi_S'] = np.empty((N_S,), dtype=np.float64)
    #     workspace['N_S'] = N_S

    if workspace['N_T'] != N_T:
        workspace['sigma2_est'] = np.zeros((N_T,), dtype=np.float64)
        workspace['correlation'] = np.zeros((N_T,), dtype=np.float64)
        workspace['power'] = np.zeros((N_T,), dtype=np.float64)
        # workspace['nttmp'] = np.empty((N_T,), dtype=np.float64)
    else:
        workspace['sigma2_est'][:] = 0
        workspace['correlation'][:] = 0
        workspace['power'][:] = 0

    workspace['N_T'] = N_T

    return workspace


func = None


def build_func():
    slices = T.cmatrix()
    S = T.cmatrix()
    envelope = T.dvector()
    ctf = T.dmatrix()
    d = T.cmatrix()
    logW_S = T.dvector()
    logW_I = T.dvector()
    logW_R = T.dvector()
    div_in = T.dscalar()
    sigma2_coloured = T.dvector()

    N_S = S.shape[0]
    N_I = ctf.shape[0]
    N_R = slices.shape[0]
    N_T = slices.shape[1]

    cproj = slices[:, np.newaxis, :] * ctf  # r * i * t
    cim = S[:, np.newaxis, :] * d  # s * i * t
    correlation_I = T.real(cproj[:, np.newaxis, :, :]) * T.real(cim) \
        + T.imag(cproj[:, np.newaxis, :, :]) * np.imag(cim)  # r * s * i * t
    power_I = T.real(cproj[:, np.newaxis, :, :]) ** 2 + T.imag(cproj[:, np.newaxis, :, :]) ** 2  # r * s * i * t

    g_I = envelope * cproj[:, np.newaxis, :, :] - cim  # r * s * i * t

    sigma2_I = T.real(g_I) ** 2 + T.imag(g_I) ** 2  # r * s * i * t

    tmp = T.sum(sigma2_I / sigma2_coloured, axis=-1)  # r * s * i

    e_I = div_in * tmp + logW_I  # r * s * i

    g_I *= ctf  # r * s * i * t

    etmp = my_logsumexp_theano(e_I)  # r * s
    e_S = etmp + logW_S  # r * s

    tmp = logW_S + logW_R[:, np.newaxis]  # r * s
    phitmp = T.exp(e_I - etmp[:, :, np.newaxis])  # r * s * i
    I_tmp = tmp[:, :, np.newaxis] + e_I

    correlation_S = T.sum(phitmp[:, :, :, np.newaxis] * correlation_I, axis=2)  # r * s * t
    power_S = T.sum(phitmp[:, :, :, np.newaxis] * power_I, axis=2)  # r * s * t
    sigma2_S = T.sum(phitmp[:, :, :, np.newaxis] * sigma2_I, axis=2)  # r * s * t
    g_S = T.sum(phitmp[:, :, :, np.newaxis] * g_I, axis=2)  # r * s * t

    etmp = my_logsumexp_theano(e_S)  # r
    e_R = etmp + logW_R  # r

    tmp = logW_R  # r
    phitmp = np.exp(e_S - etmp[:, np.newaxis])  # r * s
    S_tmp = tmp[:, np.newaxis] + e_S
    correlation_R = T.sum(phitmp[:, :, np.newaxis] * correlation_S, axis=1)  # r * t
    power_R = T.sum(phitmp[:, :, np.newaxis] * power_S, axis=1)  # r * t
    sigma2_R = T.sum(phitmp[:, :, np.newaxis] * sigma2_S, axis=1)  # r * t

    g = T.sum(phitmp[:, :, np.newaxis] * g_S, axis=1)  # r * t

    tmp = -2.0 * div_in
    nttmp = tmp * envelope / sigma2_coloured

    e = my_logsumexp_theano(e_R)
    lse_in = -e

    # Noise estimate
    phitmp = e_R - e
    R_tmp = phitmp
    phitmp = T.exp(phitmp)

    sigma2_est = T.dot(phitmp, sigma2_R)
    correlation = T.dot(phitmp, correlation_R)
    power = T.dot(phitmp, power_R)

    global func
    func = theano.function(inputs=[slices, S, envelope, ctf, d, logW_S, logW_I, logW_R, div_in, sigma2_coloured],
                           outputs=[g, I_tmp, S_tmp, R_tmp, sigma2_est, correlation, power, nttmp, lse_in, phitmp])


def doimage_RIS(slices,  # Slices of 3D volume (N_R x N_T)
                S,  # Shift operators (N_S X N_T)
                envelope,  # (Experimental) envelope (N_T)
                ctf,  # CTF operators (rotated) (N_I x N_T)
                d,  # Image data (rotated) (N_I x N_T)
                logW_S,  # Shift weights
                logW_I,  # Inplane weights
                logW_R,  # Slice weights
                sigma2,  # Inlier noise, can be a scalar or an N_T length vector
                g,  # Where to store gradient output
                workspace):

    global func
    if not func:
        build_func()

    N_S = S.shape[0]  # Number of shifts
    assert logW_S.shape[0] == N_S

    N_I = ctf.shape[0]  # Number of inplane rotations
    assert logW_I.shape[0] == N_I
    assert d.shape[0] == N_I

    N_R = slices.shape[0]  # Number of slices (projections)
    assert logW_R.shape[0] == N_R

    N_T = slices.shape[1]  # Number of (truncated) fourier coefficients
    assert S.shape[1] == N_T
    assert ctf.shape[1] == N_T
    assert d.shape[1] == N_T

    workspace = update_workspace(workspace, N_R, N_I, N_S, N_T)

    avgphi_R = np.empty((N_R,), dtype=np.float64)
    avgphi_S = np.empty((N_S,), dtype=np.float64)
    avgphi_I = np.empty((N_I,), dtype=np.float64)

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computeGrad = g is not None
    avgphi_S.fill(-np.inf)
    avgphi_I.fill(-np.inf)

    if use_whitenoise:
        sigma2_white = sigma2
        div_in = -1.0 / (2.0 * sigma2)
        sigma2_coloured = np.zeros(N_T) + 1
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        div_in = -0.5

    if use_envelope:
        assert envelope.shape[0] == N_T
    else:
        envelope = np.zeros(N_T) + 1

    if computeGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    g_tmp, I_tmp, S_tmp, R_tmp, sigma2_est_tmp, correlation_tmp, power_tmp, nttmp, lse_in, phitmp = \
        func(slices, S, envelope, ctf, d, logW_S, logW_I, logW_R, div_in, sigma2_coloured)

    for r in range(N_R):
        for s in range(N_S):
            avgphi_I = my_logaddexp(avgphi_I, I_tmp[r, s])  # i

    for r in range(N_R):
        avgphi_S = my_logaddexp(avgphi_S, S_tmp[r])  # s

    avgphi_R = R_tmp

    sigma2_est += sigma2_est_tmp
    correlation += correlation_tmp
    power += power_tmp

    if computeGrad:
        g[:] = g_tmp
        if use_envelope or not use_whitenoise:
            g *= np.outer(phitmp, nttmp)
        else:
            phitmp *= -2.0 * div_in
            g *= phitmp

    avgphi_S -= my_logsumexp(avgphi_S)
    avgphi_I -= my_logsumexp(avgphi_I)

    return lse_in, (avgphi_S[:N_S], avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace


if __name__ == '__main__':
    build_func()
