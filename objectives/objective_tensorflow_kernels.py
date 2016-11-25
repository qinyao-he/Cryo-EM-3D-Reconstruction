import math
import numpy as np

import tensorflow as tf


def my_logsumexp(a):
    shape = list(a.shape)
    shape[len(shape) - 1] = 1
    a_max = np.max(a, axis=-1)
    a_sum = np.sum(np.exp(a - a_max.reshape(tuple(shape))), axis=-1)
    return a_max + np.log(a_sum)


def my_logaddexp(a, b):
    tmp = a - b
    return np.where(tmp > 0, a + np.log1p(np.exp(-tmp)), b + np.log1p(np.exp(tmp)))


def my_logsumexp_tensorflow(a):
    a_max = tf.reduce_max(a, reduction_indices=tf.rank(a)-1, keep_dims=True)
    a_sum = tf.reduce_sum(tf.exp(a - a_max), reduction_indices=tf.rank(a)-1)
    return tf.reduce_max(a, reduction_indices=tf.rank(a)-1) + tf.log(a_sum)


def my_logaddexp_tensorflow(a, b):
    tmp = a - b
    return tf.select(tmp > 0, a + tf.log(1 + tf.exp(-tmp)), b + tf.log(1 + tf.exp(tmp)))


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
objectives = None
inputs = None


def build_func():
    slices_tensor = tf.placeholder(dtype=tf.complex64, shape=[None, None])
    S_tensor = tf.placeholder(dtype=tf.complex64, shape=[None, None])
    envelope_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
    ctf_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None])
    d_tensor = tf.placeholder(dtype=tf.complex64, shape=[None, None])
    logW_S_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
    logW_I_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
    logW_R_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
    div_in_tensor = tf.placeholder(dtype=tf.float32, shape=[])
    sigma2_coloured_tensor = tf.placeholder(dtype=tf.float32, shape=[None])

    cproj = tf.expand_dims(slices_tensor, 1) * tf.complex(ctf_tensor, tf.zeros_like(ctf_tensor)) # r * i * t
    cim = tf.expand_dims(S_tensor, 1) * d_tensor  # s * i * t
    correlation_I = tf.real(tf.expand_dims(cproj, 1)) * tf.real(cim) \
                    + tf.imag(tf.expand_dims(cproj, 1)) * tf.imag(cim)  # r * s * i * t
    power_I = tf.real(cproj) ** 2 + tf.imag(cproj) ** 2  # r * i * t

    g_I =tf.complex(envelope_tensor, tf.zeros_like(envelope_tensor)) * tf.expand_dims(cproj, 1) - cim  # r * s * i * t

    sigma2_I = tf.real(g_I) ** 2 + tf.imag(g_I) ** 2  # r * s * i * t

    tmp = tf.reduce_sum(sigma2_I / sigma2_coloured_tensor, reduction_indices=3)  # r * s * i

    e_I = div_in_tensor * tmp + logW_I_tensor  # r * s * i

    g_I *= tf.complex(ctf_tensor, tf.zeros_like(ctf_tensor))  # r * s * i * t


    etmp = my_logsumexp_tensorflow(e_I)  # r * s
    e_S = etmp + logW_S_tensor  # r * s

    tmp = logW_S_tensor + tf.expand_dims(logW_R_tensor, 1)  # r * s
    phitmp = tf.exp(e_I - tf.expand_dims(etmp, 2))  # r * s * i
    I_tmp = tf.expand_dims(tmp, 2) + e_I

    correlation_S = tf.reduce_sum(tf.expand_dims(phitmp, 3) * correlation_I, reduction_indices=2)  # r * s * t
    power_S = tf.reduce_sum(tf.expand_dims(phitmp, 3) * tf.expand_dims(power_I, 1), reduction_indices=2)  # r * s * t
    sigma2_S = tf.reduce_sum(tf.expand_dims(phitmp, 3) * sigma2_I, reduction_indices=2)  # r * s * t
    g_S = tf.reduce_sum(tf.complex(tf.expand_dims(phitmp, 3), tf.zeros_like(tf.expand_dims(phitmp, 3)))
        * g_I, reduction_indices=2)  # r * s * t

    etmp = my_logsumexp_tensorflow(e_S)  # r
    e_R = etmp + logW_R_tensor  # r

    tmp = logW_R_tensor  # r
    phitmp = tf.exp(e_S - tf.expand_dims(etmp, 1))  # r * s
    S_tmp = tf.expand_dims(tmp, 1) + e_S
    correlation_R = tf.reduce_sum(tf.expand_dims(phitmp, 2) * correlation_S, reduction_indices=1)  # r * t
    power_R = tf.reduce_sum(tf.expand_dims(phitmp, 2) * power_S, reduction_indices=1)  # r * t
    sigma2_R = tf.reduce_sum(tf.expand_dims(phitmp, 2) * sigma2_S, reduction_indices=1)  # r * t

    g = tf.reduce_sum(tf.complex(tf.expand_dims(phitmp, 2), tf.zeros_like(tf.expand_dims(phitmp, 2)))
        * g_S, reduction_indices=1)  # r * t

    tmp = -2.0 * div_in_tensor
    nttmp = tmp * envelope_tensor / sigma2_coloured_tensor

    e = my_logsumexp_tensorflow(e_R)
    lse_in = -e

    # Noise estimate
    phitmp = e_R - e  # r
    R_tmp = phitmp
    phitmp = tf.exp(phitmp)

    sigma2_est = tf.squeeze(tf.matmul(tf.expand_dims(phitmp, 0), sigma2_R), squeeze_dims=[0])
    correlation = tf.squeeze(tf.matmul(tf.expand_dims(phitmp, 0), correlation_R), squeeze_dims=[0])
    power = tf.squeeze(tf.matmul(tf.expand_dims(phitmp, 0), power_R), squeeze_dims=[0])

    global func
    global inputs
    global objectives
    func = tf.Session()
    inputs = [slices_tensor,
              S_tensor,
              envelope_tensor,
              ctf_tensor,
              d_tensor,
              logW_S_tensor,
              logW_I_tensor,
              logW_R_tensor,
              div_in_tensor,
              sigma2_coloured_tensor]
    objectives = [g, I_tmp, S_tmp, R_tmp, sigma2_est, correlation, power, nttmp, lse_in, phitmp]


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

    global inputs
    global objectives
    g_tmp, I_tmp, S_tmp, R_tmp, sigma2_est_tmp, correlation_tmp, power_tmp, nttmp, lse_in, phitmp = \
        func.run(objectives, feed_dict={
            inputs[0]: slices,
            inputs[1]: S,
            inputs[2]: envelope,
            inputs[3]: ctf,
            inputs[4]: d,
            inputs[5]: logW_S,
            inputs[6]: logW_I,
            inputs[7]: logW_R,
            inputs[8]: div_in,
            inputs[9]: sigma2_coloured
        })

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
