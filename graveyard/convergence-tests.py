import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import h5py
import hdf5plugin

import emcee

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def main():

    # reader = emcee.backends.HDFBackend("backend_long.h5", read_only=True)
    # flatchain = reader.get_chain(flat=True)
    # print(reader.get_autocorr_time())
    # return

    backend = h5py.File("backend_long.h5", "r")

    chain = backend['mcmc/chain'][:]

    # return
    y = chain[:, :, 0].T



    # Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)


    new = np.empty(len(N))
    for i, n in enumerate(N):
        new[i] = autocorr_new(y[:, :n])

    # # Plot the comparisons
    plt.style.use('science')
    plt.loglog(N, new, "o-", label="auto-corr")
    # plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14);

    plt.xlim(5e1, 3e5)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()