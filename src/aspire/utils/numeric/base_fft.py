class FFT:
    """
    Define a customized interface for FFT functions

    To make consistent among Pyfftw, Scipyfft and cupy fft,
    not all arguments are included.
    """

    def fft(self, x, axis=-1, workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def ifft(self, x, axis=-1, workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def fft2(self, x, axes=(-2, -1), workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def ifft2(self, x, axes=(-2, -1), workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def fftn(self, x, axes=None, workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def ifftn(self, x, axes=None, workers=-1):
        raise NotImplementedError("subclasses must implement this")

    def fftshift(self, x, axes=None):
        raise NotImplementedError("subclasses must implement this")

    def ifftshift(self, x, axes=None):
        raise NotImplementedError("subclasses must implement this")

    def centered_ifft(self, x, axis=-1, workers=-1):
        x = self.ifftshift(x, axes=axis)
        x = self.ifft(x, axis=axis, workers=workers)
        x = self.fftshift(x, axes=axis)
        return x

    def centered_fft(self, x, axis=-1, workers=-1):
        x = self.ifftshift(x, axes=axis)
        x = self.fft(x, axis=axis, workers=workers)
        x = self.fftshift(x, axes=axis)
        return x

    def centered_ifft2(self, x, axes=(-2, -1), workers=-1):
        x = self.ifftshift(x, axes=axes)
        x = self.ifft2(x, axes=axes, workers=workers)
        x = self.fftshift(x, axes=axes)
        return x

    def centered_fft2(self, x, axes=(-2, -1), workers=-1):
        x = self.ifftshift(x, axes=axes)
        x = self.fft2(x, axes=axes, workers=workers)
        x = self.fftshift(x, axes=axes)
        return x

    def centered_ifftn(self, x, axes=None, workers=-1):
        x = self.ifftshift(x, axes=axes)
        x = self.ifftn(x, axes=axes, workers=workers)
        x = self.fftshift(x, axes=axes)
        return x

    def centered_fftn(self, x, axes=None, workers=-1):
        x = self.ifftshift(x, axes=axes)
        x = self.fftn(x, axes=axes, workers=workers)
        x = self.fftshift(x, axes=axes)
        return x
