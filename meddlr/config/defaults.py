from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Base configurations
# -----------------------------------------------------------------------------
_MRI_RECON_TFM = CN()
_MRI_RECON_TFM.TRANSFORMS = ()
_MRI_RECON_TFM.SCHEDULER_P = CN(new_allowed=True)
_MRI_RECON_TFM.SCHEDULER_P.IGNORE = False
_MRI_RECON_TFM.AUG_SENSITIVITY_MAPS = True

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# The config version.
_C.VERSION = 1

_C.MODEL = CN()
# The device to use for the experiment.
_C.MODEL.DEVICE = "cuda"
# The name of the model. See meddlr/modeling/meta_arch for options.
_C.MODEL.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
# Path to weights to use to initialize the model.
# Useful if you want to fine tune the model.
_C.MODEL.WEIGHTS = ""

# -----------------------------------------------------------------------------
# Unrolled model config
# -----------------------------------------------------------------------------
_C.MODEL.UNROLLED = CN()
# Block type to use for the unrolled model.
_C.MODEL.UNROLLED.BLOCK_ARCHITECTURE = "ResNet"
# Number of unrolled steps.
_C.MODEL.UNROLLED.NUM_UNROLLED_STEPS = 5
# Number of residual blocks per unrolled step.
# Only applicable for ResNet block architecture.
_C.MODEL.UNROLLED.NUM_RESBLOCKS = 2
# Number of channels for ResNet conv block.
# Only applicable for ResNet block architecture.
_C.MODEL.UNROLLED.NUM_FEATURES = 256
# Dropout after each conv block.
# Only applicable for ResNet block architecture.
_C.MODEL.UNROLLED.DROPOUT = 0.0
# Padding options.
# Only applicable for ResNet block architecture.
_C.MODEL.UNROLLED.PADDING = ""
# Step size for each unrolled block.
# If only one element, the same step size is used for all blocks.
_C.MODEL.UNROLLED.STEP_SIZES = (-2.0,)
# Whether to fix the step size or make it learnable.
_C.MODEL.UNROLLED.FIX_STEP_SIZE = False
# Whether to share weights between each unrolled block.
_C.MODEL.UNROLLED.SHARE_WEIGHTS = False
# Kernel size
_C.MODEL.UNROLLED.KERNEL_SIZE = (3,)
# Number of sensitivity maps per scan. Currently only 1 is supported.
_C.MODEL.UNROLLED.NUM_EMAPS = 1

# Conv block parameters
_C.MODEL.UNROLLED.CONV_BLOCK = CN()
# Either "relu" or "leaky_relu"
_C.MODEL.UNROLLED.CONV_BLOCK.ACTIVATION = "relu"
# Normalization type. Either "none", "instance", or "batch"
_C.MODEL.UNROLLED.CONV_BLOCK.NORM = "none"
# Use affine on normalization block.
_C.MODEL.UNROLLED.CONV_BLOCK.NORM_AFFINE = False
_C.MODEL.UNROLLED.CONV_BLOCK.ORDER = ("norm", "act", "drop", "conv")

# Reconstruction loss parameters.
_C.MODEL.RECON_LOSS = CN()
# The reconstruction loss type. See :obj:`meddlr.modeling.loss_computer` for options.
_C.MODEL.RECON_LOSS.NAME = "l1"
# Whether to renormalize the predicted image before computing the loss.
_C.MODEL.RECON_LOSS.RENORMALIZE_DATA = True
# Consistency configuration. Used for Noise2Recon and VORTEX.
_C.MODEL.CONSISTENCY = CN()
# Whether to use consistency loss in the latent space.
_C.MODEL.CONSISTENCY.USE_LATENT = False
# Whether to use the consistency loss.
_C.MODEL.CONSISTENCY.USE_CONSISTENCY = True
# The loss to use for latent space consistency.
_C.MODEL.CONSISTENCY.LATENT_LOSS_NAME = "mag_l1"
_C.MODEL.CONSISTENCY.NUM_LATENT_LAYERS = 1
# The consistency loss.
_C.MODEL.CONSISTENCY.LOSS_NAME = "l1"
# The weighting for consistency loss.
_C.MODEL.CONSISTENCY.LOSS_WEIGHT = 0.1
# The weighting for latent space consistency loss.
_C.MODEL.CONSISTENCY.LATENT_LOSS_WEIGHT = 0.1
# Consistency Augmentations
_C.MODEL.CONSISTENCY.AUG = CN()
_C.MODEL.CONSISTENCY.AUG.NOISE = CN()
# Noise standard deviation - 1,5,8 used for 3D FSE in Lustig paper.
_C.MODEL.CONSISTENCY.AUG.NOISE.STD_DEV = (1,)
_C.MODEL.CONSISTENCY.AUG.MOTION = CN()
# Motion range (alpha in the VORTEX paper).
_C.MODEL.CONSISTENCY.AUG.MOTION.RANGE = (0.2, 0.5)

# Noise scheduler
_C.MODEL.CONSISTENCY.AUG.NOISE.SCHEDULER = CN()
_C.MODEL.CONSISTENCY.AUG.NOISE.SCHEDULER.WARMUP_METHOD = ""
_C.MODEL.CONSISTENCY.AUG.NOISE.SCHEDULER.WARMUP_ITERS = 0
# Noise masking
_C.MODEL.CONSISTENCY.AUG.NOISE.MASK = CN()
_C.MODEL.CONSISTENCY.AUG.NOISE.MASK.RHO = 1.0
_C.MODEL.CONSISTENCY.AUG.MRI_RECON = _MRI_RECON_TFM.clone()

# Motion scheduler
_C.MODEL.CONSISTENCY.AUG.MOTION.SCHEDULER = CN()
_C.MODEL.CONSISTENCY.AUG.MOTION.SCHEDULER.WARMUP_METHOD = ""
_C.MODEL.CONSISTENCY.AUG.MOTION.SCHEDULER.WARMUP_ITERS = 0

# Image normalization options.
# See meddlr/data/transforms/transform.py for more details.
_C.MODEL.NORMALIZER = CN()
_C.MODEL.NORMALIZER.NAME = "TopMagnitudeNormalizer"
# Keywords to normalize
# See meddlr/data/transforms/transform.py for more details
_C.MODEL.NORMALIZER.KEYWORDS = ()

_C.MODEL.SEG = CN()
_C.MODEL.SEG.CLASSES = ()
_C.MODEL.SEG.INCLUDE_BACKGROUND = False
_C.MODEL.SEG.ACTIVATION = "sigmoid"

# -----------------------------------------------------------------------------
# UNET model - shared by UNetModel and GeneralizedUNet modules
# -----------------------------------------------------------------------------
_C.MODEL.UNET = CN()
_C.MODEL.UNET.OUT_CHANNELS = 2
_C.MODEL.UNET.IN_CHANNELS = 2
_C.MODEL.UNET.CHANNELS = 32
_C.MODEL.UNET.NUM_POOL_LAYERS = 4
_C.MODEL.UNET.DROPOUT = 0.0
_C.MODEL.UNET.NORMALIZE = False
# BLOCK_ORDER only applies to GeneralizedUNet
_C.MODEL.UNET.BLOCK_ORDER = ("conv", "relu", "conv", "relu", "batchnorm", "dropout")

# -----------------------------------------------------------------------------
# Denoising model
# -----------------------------------------------------------------------------
_C.MODEL.DENOISING = CN()
_C.MODEL.DENOISING.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.DENOISING.NOISE = CN()
# Noise standard deviation to use for augmentations.
_C.MODEL.DENOISING.NOISE.STD_DEV = (1,)
# When fully sampled kspace is available, if True, perform denoising on the
# fully sampled kspace. If False, denoising will be performed on the
# randomly generated undersampled kspace.
_C.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET = True
# Same as above, but at eval time (e.g. validation).
# Defaults to :obj:`MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET`
_C.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET_EVAL = None

# -----------------------------------------------------------------------------
# Compressed Sensing (CS) model
# -----------------------------------------------------------------------------
_C.MODEL.CS = CN()
_C.MODEL.CS.REGULARIZATION = 0.005
_C.MODEL.CS.MAX_ITER = 200

# -----------------------------------------------------------------------------
# Noise2Recon (N2R) model
# -----------------------------------------------------------------------------
_C.MODEL.N2R = CN()
_C.MODEL.N2R.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.N2R.USE_SUPERVISED_CONSISTENCY = False
# Use :obj:`MODEL.CONSISTENCY.AUG.NOISE.STD_DEV` to control noise parameters.

# -----------------------------------------------------------------------------
# Motion2Recon (M2R) model
# -----------------------------------------------------------------------------
_C.MODEL.M2R = CN()
_C.MODEL.M2R.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.M2R.USE_SUPERVISED_CONSISTENCY = False
# Use :obj:`MODEL.CONSISTENCY.AUG.MOTION_RANGE` to control motion parameters.

# -----------------------------------------------------------------------------
# NoiseMotion2Recon (NM2R) model
# -----------------------------------------------------------------------------
_C.MODEL.NM2R = CN()
_C.MODEL.NM2R.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.NM2R.USE_SUPERVISED_CONSISTENCY = False
# Use MODEL.CONSISTENCY.AUG.MOTION_RANGE to control motion parameters.

# -----------------------------------------------------------------------------
# Aug2Recon (A2R) / VORTEX model
# -----------------------------------------------------------------------------
_C.MODEL.A2R = CN()
_C.MODEL.A2R.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.A2R.USE_SUPERVISED_CONSISTENCY = False

# -----------------------------------------------------------------------------
# SSDU model
# -----------------------------------------------------------------------------
_C.MODEL.SSDU = CN()
_C.MODEL.SSDU.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.SSDU.MASKER = CN()
_C.MODEL.SSDU.MASKER.PARAMS = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
# List of the dataset names for validation. Must be registered in DatasetCatalog
_C.DATASETS.VAL = ()
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of batches to prefetch per worker
_C.DATALOADER.PREFETCH_FACTOR = 2
# If True, the dataloader will drop the last batch.
_C.DATALOADER.DROP_LAST = True
# Subsample training data to simulate data limited scenarios.
_C.DATALOADER.SUBSAMPLE_TRAIN = CN()
# Number of training examples to retain. All others will be force dropped
# meaning they will not be used for any training, even w/ semi-supervised N2R
# framework
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL = -1
# Number of training examples to retain based on metadata properties.
# e.g. (("acquisition", {"AXT2": 15, "AXT1": 10}),) will only keep
# 15 AXT2 scans and 10 AXT1 scans where AXT2 and AXT1 are categories in the
# "acquisition" metadata field.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP = ()
# Number of scans out of total to undersample. If NUM_TOTAL is not -1, must be
# less than NUM_TOTAL.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED = 0
# Number of scans with ground truth for validation.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL = -1
# Number of validation examples to retain based on metadata properties.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP = ()
# Seed for shuffling data. Should always be deterministic
_C.DATALOADER.SUBSAMPLE_TRAIN.SEED = 1000
# Options: "" (defaults to random sampling), "AlternatingSampler"
_C.DATALOADER.SAMPLER_TRAIN = ""
# AlternatingSampler config parameters.
_C.DATALOADER.ALT_SAMPLER = CN()
_C.DATALOADER.ALT_SAMPLER.PERIOD_SUPERVISED = 1
_C.DATALOADER.ALT_SAMPLER.PERIOD_UNSUPERVISED = 1
# GroupSampler config parameters.
_C.DATALOADER.GROUP_SAMPLER = CN()
_C.DATALOADER.GROUP_SAMPLER.BATCH_BY = ()
_C.DATALOADER.GROUP_SAMPLER.AS_BATCH_SAMPLER = False
# Paired tuple of data keys and H5DF keys. Empty tuple will result in default keys being used.
# e.g. (("target", "espirit_recon"), ("maps", "espirit_maps"))
# See data/slice_data.py for more information.
_C.DATALOADER.DATA_KEYS = ()

_C.DATALOADER.FILTER = CN()
# Paired tuple of key and values for filtering. Multiple values should be specified as tuple.
# e.g. (("num_slices", 30),) will only keep data with number of slices = 30
# Field must appear in all dataset dicts.
# Training, validation, and testing data is filtered by this field.
_C.DATALOADER.FILTER.BY = ()


# -----------------------------------------------------------------------------
# Augmentations/Transforms
# -----------------------------------------------------------------------------
_C.AUG_TRAIN = CN()
_C.AUG_TRAIN.UNDERSAMPLE = CN()
_C.AUG_TRAIN.UNDERSAMPLE.NAME = "PoissonDiskMaskFunc"
_C.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (6,)
_C.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE = 20
_C.AUG_TRAIN.UNDERSAMPLE.CENTER_FRACTIONS = ()
_C.AUG_TRAIN.UNDERSAMPLE.MAX_ATTEMPTS = 30
# Augment with noise. See MODEL.CONSISTENCY.AUG.NOISE for parameters.
_C.AUG_TRAIN.USE_NOISE = False
_C.AUG_TRAIN.USE_MOTION = False
_C.AUG_TRAIN.NOISE_P = 0.2
_C.AUG_TRAIN.MOTION_P = 0.2
# Augmentations using MRIReconAugmentor pipeline.
# If this is specified, it will be used in place of MOTION and NOISE
# augmentation parameters above. If you would like to use those parameters
# set AUG_TRAIN.MRI_RECON.TRANSFORMS=().
# This augmentation pipeline will automatically be ignored at test time (for now).
_C.AUG_TRAIN.MRI_RECON = _MRI_RECON_TFM.clone()

_C.AUG_TEST = CN()
_C.AUG_TEST.UNDERSAMPLE = CN()
_C.AUG_TEST.UNDERSAMPLE.ACCELERATIONS = (6,)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER = "Adam"

# See meddlr/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 20

_C.SOLVER.BASE_LR = 1e-4

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GRAD_ACCUM_ITERS = 1

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 1

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.TRAIN_BATCH_SIZE = 16
_C.SOLVER.TEST_BATCH_SIZE = 16

_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# The period over which to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 1
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# Validate with test-like functionality.
# If True, undersampling masks for every validation scan will be fixed
# given an acceleration.
_C.TEST.VAL_AS_TEST = True
# Validation metrics.
_C.TEST.VAL_METRICS = CN()
_C.TEST.VAL_METRICS.RECON = ()
# The period over which to flush evaluation results.
# Specify if you are getting OOM errors during evaluation.
# Set to 0 to disable.
_C.TEST.FLUSH_PERIOD = 0
_C.TEST.POSTPROCESSOR = CN()
_C.TEST.POSTPROCESSOR.NAME = ""  # e.g. "hard_dc" for hard data consistency

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = ""
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable. Currently not functional.
# This will always be in number of iterations.
_C.VIS_PERIOD = 0
# The scale when referring to time generally in the config.
# Note that there are certain fields, which are explicitly at the iteration
# scale (e.g. GRAD_ACCUM_ITERS).
# Either "epoch" or "iter"
_C.TIME_SCALE = "epoch"
_C.CUDNN_BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Config Description
# ---------------------------------------------------------------------------- #
_C.DESCRIPTION = CN()
# Brief description about config
_C.DESCRIPTION.BRIEF = ""
# The entity (team/individual) account for logging to Weights & Biases
_C.DESCRIPTION.ENTITY_NAME = "ss_recon"
# Project name for logging to Weights & Biases
_C.DESCRIPTION.PROJECT_NAME = "ss_recon"
# Experiment name for logging to Weights & Biases
_C.DESCRIPTION.EXP_NAME = ""
# Tags associated with experiment.
# e.g. "fastmri_knee_mc" for fastMRI dataset; "unrolled" for using unrolled network; etc.
_C.DESCRIPTION.TAGS = ()
