"""Convenient functions for orthonormal centered FFT"""
try:
    import pyfftw.interfaces.numpy_fft as fft
except BaseException:
    from numpy import fft


def ifftnc(x, axes, ortho=True):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.ifftn(tmp, axes=axes, norm="ortho" if ortho else None)
    return fft.ifftshift(tmp, axes=axes)


def fftnc(x, axes, ortho=True):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.fftn(tmp, axes=axes, norm="ortho" if ortho else None)
    return fft.ifftshift(tmp, axes=axes)


def fftc(x, axis=0, ortho=True):
    return fftnc(x, (axis,), ortho=ortho)


def ifftc(x, axis=0, ortho=True):
    return ifftnc(x, (axis,), ortho=ortho)


def fft2c(x, ortho=True):
    return fftnc(x, (-2, -1), ortho=ortho)


def ifft2c(x, ortho=True):
    return ifftnc(x, (-2, -1), ortho=ortho)


def fft3c(x, ortho=True):
    return fftnc(x, (-3, -2, -1), ortho=ortho)


def ifft3c(x, ortho=True):
    return ifftnc(x, (-3, -2, -1), ortho=ortho)
