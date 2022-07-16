.. _api_metrics:

meddlr.metrics
==============

Meddlr metrics extend the `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ interface
to support tracking and reporting metrics on a per-example, per-channel basis. This can be useful
for more nuanced model monitoring, such as determining which set of examples is the model underperforming
on or which categories have the least accuracy.

Because these metrics are based on the torchmetrics interface, they are compatible everywhere where torchmetrics
metrics are used.


Functional
^^^^^^^^^^
All metrics in Meddlr are also available as functions in the :mod:`meddlr.metrics.functional` module.

.. autosummary::
   :toctree: generated
   :nosignatures:

   meddlr.metrics.functional.mse
   meddlr.metrics.functional.nrmse
   meddlr.metrics.functional.psnr
   meddlr.metrics.functional.rmse
   meddlr.metrics.functional.ssim
   meddlr.metrics.functional.assd
   meddlr.metrics.functional.cv
   meddlr.metrics.functional.dice
   meddlr.metrics.functional.voe
