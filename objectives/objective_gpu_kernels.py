import math
import numpy as np
import theano


def my_logsumexp(a):
    a_max = np.max(a)
    a_sum = np.sum(np.exp(a - a_max))
    return a_max + math.log(a_sum)


def my_logaddexp(a, b):
    if a == b:
        return a + 0.69314718055994529  # This is the numerical value of ln(2)
    else:
        tmp = a - b

        if tmp > 0:
            return a + math.log1p(math.exp(-tmp))
        elif tmp <= 0:
            return b + math.log1p(math.exp(tmp))
        else:
            return tmp


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


def doimage_RS(slices,  # Slices of 3D volume (N_R x N_T)
               S,  # Shift operators (N_S X N_T)
               envelope,  # (Experimental) envelope (N_T)
               ctf,  # CTF (N_T)
               d,  # Image data (N_T)
               logW_S,  # Shift weights
               logW_R,  # Slice weights
               sigma2,  # Inlier noise, can be a scalar or an N_T length vector
               g,  # Where to store gradient output
               workspace):  # a workspace dictionary

    N_S = S.shape[0]  # Number of shifts
    assert logW_S.shape[0] == N_S

    N_R = slices.shape[0]  # Number of slices (projections)
    assert logW_R.shape[0] == N_R

    N_T = slices.shape[1]  # Number of (truncated) fourier coefficients
    assert S.shape[1] == N_T
    assert ctf.shape[0] == N_T
    assert d.shape[0] == N_T

    workspace = update_workspace(workspace, N_R, None, N_S, N_T)

    g_S = workspace['g_S']

    e_R = workspace['e_R']
    sigma2_R = workspace['sigma2_R']
    correlation_R = workspace['correlation_R']
    power_R = workspace['power_R']
    avgphi_R = workspace['avgphi_R']

    e_S = workspace['e_S']
    sigma2_S = workspace['sigma2_S']
    correlation_S = workspace['correlation_S']
    power_S = workspace['power_S']
    avgphi_S = workspace['avgphi_S']

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    nttmp = workspace['nttmp']

    use_envelope = envelope is not None
    use_whitenoise = not isinstance(sigma2, np.ndarray)
    computeGrad = g is not None
    avgphi_S.fill(-np.inf)

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
            cproj = ctf * slices[r, :]
            cim = S[s, :] * d
            correlation_S[s, :] = np.real(cproj) * np.real(cim) + np.imag(cproj) * np.imag(cim)
            if use_envelope:
                g_S[s, :] = envelope * cproj - cim
            else:
                g_S[s, :] = cproj - cim

            # Compute the log likelihood
            if use_whitenoise:
                sigma2_S[s, :] = np.real(g_S[s, :]) ** 2 + np.imag(g_S[s, :]) ** 2
                tmp = np.sum(sigma2_S[s, :])
            else:
                sigma2_S[s, :] = np.real(g_S[s, :]) ** 2 + np.imag(g_S[s, :]) ** 2
                tmp = np.sum(sigma2_S[s, :] / sigma2_coloured)

            e_S[s] = div_in * tmp + logW_S[s]

        etmp = my_logsumexp(e_S)
        e_R[r] = etmp + logW_R[r]

        # Noise estimate
        sigma2_R[r, :].fill(0)
        correlation_R[r, :].fill(0)
        power_R[r, :].fill(0)
        tmp = logW_R[r]
        for s in range(N_S):
            phitmp = math.exp(e_S[s] - etmp)
            avgphi_S[s] = my_logaddexp(avgphi_S[s], tmp + e_S[s])
            for t in xrange(N_T):
                correlation_R[r, t] += phitmp * correlation_S[s, t]
            for t in xrange(N_T):
                power_R[r, t] += phitmp * power_S[s, t]
            for t in xrange(N_T):
                sigma2_R[r, t] += phitmp * sigma2_S[s, t]

            if computeGrad:
                for t in xrange(N_T):
                    g[r, t] = g[r, t] + phitmp * g_S[s, t]

    if computeGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                for t in xrange(N_T):
                    nttmp[t] = tmp * (ctf[t] * envelope[t]) / sigma2_coloured[t]
            else:
                for t in xrange(N_T):
                    nttmp[t] = tmp * ctf[t] / sigma2_coloured[t]
        else:
            if use_envelope:
                for t in xrange(N_T):
                    nttmp[t] = tmp * (ctf[t] * envelope[t])
            else:
                for t in xrange(N_T):
                    nttmp[t] = tmp * ctf[t]

    e = my_logsumexp(e_R)
    lse_in = -e
    # Noise estimate
    for r in xrange(N_R):
        phitmp = e_R[r] - e
        avgphi_R[r] = phitmp
        phitmp = math.exp(phitmp)
        for t in xrange(N_T):
            sigma2_est[t] += phitmp * sigma2_R[r, t]
        for t in xrange(N_T):
            correlation[t] += phitmp * correlation_R[r, t]
        for t in xrange(N_T):
            power[t] += phitmp * power_R[r, t]

        if computeGrad:
            for t in xrange(N_T):
                g[r, t] *= phitmp * nttmp[t]

    avgphi_S -= my_logsumexp(avgphi_S)

    return lse_in, (avgphi_S[:N_S], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace


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

    workspace = update_workspace(workspace, N_R, N_I, N_S, N_T)

    g_I = workspace['g_I']
    g_S = workspace['g_S']

    e_R = workspace['e_R']
    sigma2_R = workspace['sigma2_R']
    correlation_R = workspace['correlation_R']
    power_R = workspace['power_R']
    avgphi_R = workspace['avgphi_R']

    e_I = workspace['e_I']
    sigma2_I = workspace['sigma2_I']
    correlation_I = workspace['correlation_I']
    power_I = workspace['power_I']
    avgphi_I = workspace['avgphi_I']

    e_S = workspace['e_S']
    sigma2_S = workspace['sigma2_S']
    correlation_S = workspace['correlation_S']
    power_S = workspace['power_S']
    avgphi_S = workspace['avgphi_S']

    sigma2_est = workspace['sigma2_est']
    correlation = workspace['correlation']
    power = workspace['power']

    nttmp = workspace['nttmp']

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
            for i in xrange(N_I):
                # Compute the error at each frequency
                for t in xrange(N_T):
                    cproj = ctf[i, t] * slices[r, t]
                    cim = S[s, t] * d[i, t]

                    correlation_I[i, t] = cproj.real * cim.real + cproj.imag * cim.imag
                    power_I[i, t] = cproj.real * cproj.real + cproj.imag * cproj.imag

                    if use_envelope:
                        g_I[i, t] = envelope[t] * cproj - cim
                    else:
                        g_I[i, t] = cproj - cim

                # Compute the log likelihood
                tmp = 0
                if use_whitenoise:
                    for t in xrange(N_T):
                        sigma2_I[i, t] = g_I[i, t].real ** 2 + g_I[i, t].imag ** 2
                        tmp += sigma2_I[i, t]
                else:
                    for t in xrange(N_T):
                        sigma2_I[i, t] = g_I[i, t].real ** 2 + g_I[i, t].imag ** 2
                        tmp += sigma2_I[i, t] / sigma2_coloured[t]
                e_I[i] = div_in * tmp + logW_I[i]

                # Compute the gradient
                if computeGrad:
                    for t in xrange(N_T):
                        g_I[i, t] = ctf[i, t] * g_I[i, t]
                        # Since the envelope and sigma2_coloured don't depend
                        # on r, i or s (just on t), we can multiply the gradient
                        # at the end.

            etmp = my_logsumexp(e_I)
            e_S[s] = etmp + logW_S[s]

            # Noise estimate
            for t in xrange(N_T):
                sigma2_S[s, t] = 0
                correlation_S[s, t] = 0
                power_S[s, t] = 0
            if computeGrad:
                for t in range(N_T):
                    g_S[s, t] = 0
            tmp = logW_S[s] + logW_R[r]
            for i in xrange(N_I):
                phitmp = math.exp(e_I[i] - etmp)
                avgphi_I[i] = my_logaddexp(avgphi_I[i], tmp + e_I[i])
                for t in xrange(N_T):
                    correlation_S[s, t] += phitmp * correlation_I[i, t]
                    power_S[s, t] += phitmp * power_I[i, t]
                    sigma2_S[s, t] += phitmp * sigma2_I[i, t]

                if computeGrad:
                    for t in range(N_T):
                        g_S[s, t] = g_S[s, t] + phitmp * g_I[i, t]

        etmp = my_logsumexp(e_S)
        e_R[r] = etmp + logW_R[r]

        # Noise estimate
        for t in xrange(N_T):
            sigma2_R[r, t] = 0
            correlation_R[r, t] = 0
            power_R[r, t] = 0
        tmp = logW_R[r]
        for s in range(N_S):
            phitmp = math.exp(e_S[s] - etmp)
            avgphi_S[s] = my_logaddexp(avgphi_S[s], tmp + e_S[s])
            for t in xrange(N_T):
                correlation_R[r, t] += phitmp * correlation_S[s, t]
            for t in xrange(N_T):
                power_R[r, t] += phitmp * power_S[s, t]
            for t in xrange(N_T):
                sigma2_R[r, t] += phitmp * sigma2_S[s, t]

            if computeGrad:
                for t in xrange(N_T):
                    g[r, t] = g[r, t] + phitmp * g_S[s, t]

    e = my_logsumexp(e_R)
    lse_in = -e

    if computeGrad:
        tmp = -2.0 * div_in
        if not use_whitenoise:
            if use_envelope:
                for t in xrange(N_T):
                    nttmp[t] = tmp * envelope[t] / sigma2_coloured[t]
            else:
                for t in xrange(N_T):
                    nttmp[t] = tmp / sigma2_coloured[t]
        else:
            if use_envelope:
                for t in xrange(N_T):
                    nttmp[t] = tmp * envelope[t]

    # Noise estimate
    for r in xrange(N_R):
        phitmp = e_R[r] - e
        avgphi_R[r] = phitmp
        phitmp = math.exp(phitmp)
        for t in xrange(N_T):
            sigma2_est[t] += phitmp * sigma2_R[r, t]
        for t in xrange(N_T):
            correlation[t] += phitmp * correlation_R[r, t]
        for t in xrange(N_T):
            power[t] += phitmp * power_R[r, t]

        if computeGrad:
            if use_envelope or not use_whitenoise:
                for t in xrange(N_T):
                    g[r, t] = phitmp * nttmp[t] * g[r, t]
            else:
                phitmp *= -2.0 * div_in
                for t in xrange(N_T):
                    g[r, t] = phitmp * g[r, t]

    es = my_logsumexp(avgphi_S)
    for s in xrange(N_S):
        avgphi_S[s] = avgphi_S[s] - es
    ei = my_logsumexp(avgphi_I)
    for i in xrange(N_I):
        avgphi_I[i] = avgphi_I[i] - ei

    return lse_in, (avgphi_S[:N_S], avgphi_I[:N_I], avgphi_R[:N_R]), sigma2_est, correlation, power, workspace
