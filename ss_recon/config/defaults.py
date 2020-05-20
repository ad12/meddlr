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
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.VERSION = 1

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.WEIGHTS = ""

# -----------------------------------------------------------------------------
# Unrolled model - TODO: Deprecate
# -----------------------------------------------------------------------------
_C.MODEL.UNROLLED = CN()
_C.MODEL.UNROLLED.NUM_UNROLLED_STEPS = 5
_C.MODEL.UNROLLED.NUM_RESBLOCKS = 2
_C.MODEL.UNROLLED.NUM_FEATURES = 256
_C.MODEL.UNROLLED.DROPOUT = 0.0
# Padding options. "" or "circular"
_C.MODEL.UNROLLED.PADDING = ""
_C.MODEL.UNROLLED.FIX_STEP_SIZE = False
_C.MODEL.UNROLLED.SHARE_WEIGHTS = False
# Kernel size
_C.MODEL.UNROLLED.KERNEL_SIZE = (3,)
# Number of ESPIRiT maps
_C.MODEL.UNROLLED.NUM_EMAPS = 1

_C.MODEL.RECON_LOSS = CN()
_C.MODEL.RECON_LOSS.NAME = "l1"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If True, the dataloader will drop the last batch.
_C.DATALOADER.DROP_LAST = True
# Subsample training data to simulate data limited scenarios.
_C.DATALOADER.SUBSAMPLE_TRAIN = CN()
# Number of training examples to retain. All others will be force dropped
# meaning they will not be used for any training, even w/ semi-supervised N2R
# framework
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL = -1
# Number of scans out of total to undersample. If NUM_TOTAL is not -1, must be
# less than NUM_TOTAL.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED = 0
# Seed for shuffling data. Should always be deterministic
_C.DATALOADER.SUBSAMPLE_TRAIN.SEED = 1000


# -----------------------------------------------------------------------------
# Augmentations/Transforms
# -----------------------------------------------------------------------------
_C.AUG_TRAIN = CN()
_C.AUG_TRAIN.UNDERSAMPLE = CN()
_C.AUG_TRAIN.UNDERSAMPLE.NAME = "RandomMaskFunc"
_C.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (6, 8)
_C.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE = 20

_C.AUG_TEST = CN()
_C.AUG_TEST.UNDERSAMPLE = CN()
_C.AUG_TEST.UNDERSAMPLE.ACCELERATIONS = (6, 8)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER = "Adam"

# See ss_recon/solver/build.py for LR scheduler options
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

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 1

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.TRAIN_BATCH_SIZE = 16
_C.SOLVER.TEST_BATCH_SIZE = 16

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
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
_C.VIS_PERIOD = 0
# The scale when referring to time generally in the config.
# Note that there are certain fields, which are explicitly at the iteration
# scale (e.g. NUM_GRAD_STEPS).
# Either "epoch" or "iter"
_C.TIME_SCALE = "epoch"
_C.CUDNN_BENCHMARK = False
