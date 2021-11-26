.. _api_ops:

meddlr.ops
==============
Meddlr provides some utilities for complex, categorical, and fft-related operations.
All operations assume the first dimension is the batch dimension.


Complex Utilities
^^^^^^^^^^^^^^^^^^
These utilities are used to perform complex arithmetic with both complex tensors
and real views of compex tensors, where the last dimension is ``2`` for real/imaginary
components. Because these utilities are optimized for complex numbers, they will fail
on real-valued tensors that do not follow the real-view convention described above.

Note, because these utilites support real-view tensors, if the last dimension
is ``2``, it will be interpreted as a real-view of a complex tensor.
In your code, we recommend using complex tensors when handling complex numbers
to avoid ambiguity between real and complex numbers.

.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.ops.complex.abs
    meddlr.ops.complex.angle
    meddlr.ops.complex.center_crop
    meddlr.ops.complex.channels_first
    meddlr.ops.complex.channels_last
    meddlr.ops.complex.conj
    meddlr.ops.complex.from_polar
    meddlr.ops.complex.get_mask
    meddlr.ops.complex.imag
    meddlr.ops.complex.is_complex
    meddlr.ops.complex.is_complex_as_real
    meddlr.ops.complex.matmul
    meddlr.ops.complex.mul
    meddlr.ops.complex.power_method
    meddlr.ops.complex.real
    meddlr.ops.complex.rss
    meddlr.ops.complex.svd
    meddlr.ops.complex.to_numpy
    meddlr.ops.complex.to_tensor


Categorical Utilities
^^^^^^^^^^^^^^^^^^^^^
These utilities are designed to perform categorical operations on tensors, such as
converting between categorical and one-hot representations and converting from logits
to probabilities.


.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.ops.categorical_to_one_hot
    meddlr.ops.logits_to_prob
    meddlr.ops.one_hot_to_categorical
    meddlr.ops.pred_to_categorical

Fourier Transform Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:

    meddlr.ops.fft2c
    meddlr.ops.fft3c
    meddlr.ops.fftc
    meddlr.ops.fftnc
    meddlr.ops.fftshift
    meddlr.ops.ifft2c
    meddlr.ops.ifft3c
    meddlr.ops.ifftc
    meddlr.ops.ifftnc
    meddlr.ops.ifftshift