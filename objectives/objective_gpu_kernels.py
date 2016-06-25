import math
import numpy as np
import theano


def my_logsumexp(a):
    shape = list(a.shape)
    shape[len(shape)-1] = 1
    a_max = np.max(a, axis=-1)
    a_sum = np.sum(np.exp(a - a_max.reshape(tuple(shape))), axis=-1)
    return a_max + np.log(a_sum)


def my_logaddexp(a, b):
    tmp = a - b
    return np.select([a == b, tmp > 0, tmp <= 0], [
        a + 0.69314718055994529,
        a + np.log1p(np.exp(-tmp)),
        b + np.log1p(np.exp(tmp))
    ], default=tmp)


def update_workspace(workspace, N_R, N_I, N_S, N_T):
    if workspace is None:
        workspace = {'N_R': 0, 'N_I': 0, 'N_S': 0, 'N_T': 0}

    if N_R is not None and workspace['N_R'] < N_R or workspace['N_T'] != N_T:
        workspace['sigma2_R'] = np.empty((N_R, N_T), dtype=np.float64)
        workspace['correlation_R'] = np.empty((N_R, N_T), dtype=np.float64)
        workspace['power_R'] = np.empty((N_R, N_T), dtype=np.float64)
        if workspace['N_R'] < N_R:
            workspace['e_R'] = np.empty((N_R,), dtype=np.float64)
            workspace['avgphi_R'] = np.empty((N_R,), dtype=np.float64)
        workspace['N_R'] = N_R

    if N_I is not None and (workspace['N_I'] < N_I or workspace['N_T'] != N_T):
        workspace['sigma2_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['correlation_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['power_I'] = np.empty((N_I, N_T), dtype=np.float64)
        workspace['g_I'] = np.empty((N_I, N_T), dtype=np.complex64)
        if workspace['N_I'] < N_I:
            workspace['e_I'] = np.empty((N_I,), dtype=np.float64)
            workspace['avgphi_I'] = np.empty((N_I,), dtype=np.float64)
        workspace['N_I'] = N_I

    if N_S is not None and (workspace['N_S'] < N_S or workspace['N_T'] != N_T):
        workspace['sigma2_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['correlation_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['power_S'] = np.empty((N_S, N_T), dtype=np.float64)
        workspace['g_S'] = np.empty((N_S, N_T), dtype=np.complex64)
        if workspace['N_S'] < N_S:
            workspace['e_S'] = np.empty((N_S,), dtype=np.float64)
            workspace['avgphi_S'] = np.empty((N_S,), dtype=np.float64)
        workspace['N_S'] = N_S

    if workspace['N_T'] != N_T:
        workspace['sigma2_est'] = np.zeros((N_T,), dtype=np.float64)
        workspace['correlation'] = np.zeros((N_T,), dtype=np.float64)
        workspace['power'] = np.zeros((N_T,), dtype=np.float64)
        workspace['nttmp'] = np.empty((N_T,), dtype=np.float64)
    else:
        workspace['sigma2_est'][:] = 0
        workspace['correlation'][:] = 0
        workspace['power'][:] = 0

    workspace['N_T'] = N_T

    return workspace


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

    # workspace = update_workspace(workspace, N_R, N_I, N_S, N_T)

    g_I = np.empty((N_I, N_T), dtype=np.complex64)
    g_S = np.empty((N_S, N_T), dtype=np.complex64)

    e_R = np.empty((N_R,), dtype=np.float64)
    sigma2_R = np.empty((N_R, N_T), dtype=np.float64)
    correlation_R = np.empty((N_R, N_T), dtype=np.float64)
    power_R = np.empty((N_R, N_T), dtype=np.float64)
    avgphi_R = np.empty((N_R,), dtype=np.float64)

    e_I = np.empty((N_I,), dtype=np.float64)
    sigma2_I = np.empty((N_I, N_T), dtype=np.float64)
    correlation_I = np.empty((N_I, N_T), dtype=np.float64)
    power_I = np.empty((N_I, N_T), dtype=np.float64)
    avgphi_I = np.empty((N_I,), dtype=np.float64)

    e_S = np.empty((N_S,), dtype=np.float64)
    sigma2_S = np.empty((N_S, N_T), dtype=np.float64)
    correlation_S = np.empty((N_S, N_T), dtype=np.float64)
    power_S = np.empty((N_S, N_T), dtype=np.float64)
    avgphi_S = np.empty((N_S,), dtype=np.float64)

    sigma2_est = np.zeros((N_T,), dtype=np.float64)
    correlation = np.zeros((N_T,), dtype=np.float64)
    power = np.zeros((N_T,), dtype=np.float64)

    nttmp = np.empty((N_T,), dtype=np.float64)

    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computeGrad = g is not None
    avgphi_S.fill(-np.inf)
    avgphi_I.fill(-np.inf)

    if use_whitenoise:
        sigma2_white = sigma2
        div_in = -1.0 / (2.0 * sigma2)
    else:
        sigma2_coloured = sigma2
        assert sigma2_coloured.shape[0] == N_T
        div_in = -0.5

    if use_envelope:
        assert envelope.shape[0] == N_T

    if computeGrad:
        assert g.shape[0] == N_R
        assert g.shape[1] == N_T

    for r in xrange(N_R):
        for s in xrange(N_S):
            # Compute the error at each frequency
            cproj = ctf * slices[r, :][np.newaxis, :]
            cim = S[s, :][np.newaxis, :] * d
            correlation_I[:] = np.real(cproj) * np.real(cim) + np.imag(cproj) * np.imag(cim)
            power_I[:] = np.real(cproj) ** 2 + np.imag(cproj) ** 2
            if use_envelope:
                g_I[:] = envelope[np.newaxis, :] * cproj - cim
            else:
                g_I[:] = cproj - cim

            # Compute the log likelihood
            if use_whitenoise:
                sigma2_I[:] = np.real(g_I) ** 2 + np.imag(g_I) ** 2
                tmp = np.sum(sigma2_I, axis=1)
            else:
                sigma2_I[:] = np.real(g_I) ** 2 + np.imag(g_I) ** 2
                tmp = np.sum(sigma2_I / sigma2_coloured[np.newaxis, :], axis=1)

            e_I = div_in * tmp + logW_I

            # Compute the gradient
            if computeGrad:
                g_I *= ctf

            etmp = my_logsumexp(e_I)
            e_S[s] = etmp + logW_S[s]

            # Noise estimate
            tmp = logW_S[s] + logW_R[r]
            phitmp = np.exp(e_I - etmp)
            avgphi_I[:] = my_logaddexp(avgphi_I, tmp + e_I)
            correlation_S[s, :] = np.dot(phitmp, correlation_I)
            power_S[s, :] = np.dot(phitmp, power_I)
            sigma2_S[s, :] = np.dot(phitmp, sigma2_I)
            if computeGrad:
                g_S[s, :] = np.dot(phitmp, g_I)

        etmp = my_logsumexp(e_S)
        e_R[r] = etmp + logW_R[r]

        # Noise estimate
        tmp = logW_R[r]

        phitmp = np.exp(e_S - etmp)
        avgphi_S[:] = my_logaddexp(avgphi_S, tmp + e_S)
        correlation_R[r, :] = np.dot(phitmp, correlation_S)
        power_R[r, :] = np.dot(phitmp, power_S)
        sigma2_R[r, :] = np.dot(phitmp, sigma2_S)

        if computeGrad:
            g[r, :] = np.dot(phitmp, g_S)

    e = my_logsumexp(e_R)
    lse_in = -e

    if computeGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                nttmp[:] = tmp * envelope / sigma2_coloured
            else:
                nttmp[:] = tmp / sigma2_coloured
        else:
            if use_envelope:
                nttmp[:] = tmp * envelope

    # Noise estimate
    phitmp = e_R - e
    avgphi_R[:] = phitmp
    phitmp = np.exp(phitmp)

    sigma2_est += np.dot(phitmp, sigma2_R)
    correlation += np.dot(phitmp, correlation_R)
    power += np.dot(phitmp, power_R)

    if computeGrad:
        if use_envelope or not use_whitenoise:
            g *= np.outer(phitmp, nttmp)
        else:
            phitmp *= -2.0 * div_in
            g *= phitmp

    avgphi_S -= my_logsumexp(avgphi_S)
    avgphi_I -= my_logsumexp(avgphi_I)

    return lse_in, (avgphi_S[:N_S], avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace
