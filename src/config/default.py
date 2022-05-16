from yacs.config import CfgNode as CN
_CN = CN()


##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adam"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'constant'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

_CN.TRAINER.NUM_FRAME = 5
_CN.TRAINER.SDF_CLAMP = None
_CN.TRAINER.EIKONAL_LOSS = 0.1
_CN.TRAINER.TRACK_RUNNING_STATS = True
_CN.TRAINER.NOISE_POSE_STD = 0.01
_CN.TRAINER.NOISE_POSE_MULTIPLIER = 3
_CN.TRAINER.TRAIN_SHAPE = True
_CN.TRAINER.TRAIN_POSE = False
_CN.TRAINER.TRAIN_SCALE = False
_CN.TRAINER.TRAIN_MASK = False
_CN.TRAINER.USE_OCC = False
_CN.TRAINER.NUM_3D_POINTS = 2048
_CN.TRAINER.RIGID_ALIGN_TO_GT = False
_CN.TRAINER.LIMIT_TEST_NUMBER = 2000
_CN.TRAINER.BBOX_SCALE_MIN = 0.3
_CN.TRAINER.BBOX_SCALE_MAX = 3.5
_CN.TRAINER.BBOX_SCALE_NUM = 128
_CN.TRAINER.LOSS_BBOX_SCALE = 0.1

_CN.TRAINER.SEED = 66
_CN.TRAINER.MONITOR_KEY = 'PseudoIoU_mean'
_CN.TRAINER.MODEL = 'MODEL'
_CN.TRAINER.LIMIT_TEST_NUMBER = -1
_CN.TRAINER.BATCH_SIZE = -1
_CN.TRAINER.NORMALIZE_NORMAL = True

_CN.TRAINER.CAMERA_POSE = 'gt' # ['gt', 'gt+noise', 'predicted']

_CN.TRAINER.EVAL_PER_JOINT_STEP = False
_CN.TRAINER.BBOX = 'gt'
_CN.TRAINER.VOLUME_REDUCTION = 'mean'


_CN.DATASET = CN()
_CN.DATASET.name = 'shapenet'
_CN.DATASET.DATA_VERSION = ''
_CN.DATASET.LOAD_DEPTH = True
_CN.DATASET.SHAPENET_CATEGORY_TRAIN =  ''
_CN.DATASET.SHAPENET_CATEGORY_TEST = ''
_CN.DATASET.IMAGE_WIDTH = 224
_CN.DATASET.IMAGE_HEIGHT = 224
_CN.DATASET.DATA_SOURCE = ""
_CN.DATASET.DATA_ROOT = ""
_CN.DATASET.NPZ_ROOT = ""
_CN.DATASET.TRAIN_LIST_PATH = ""
_CN.DATASET.VAL_LIST_PATH = ""
_CN.DATASET.TEST_LIST_PATH = ""

_CN.BACKBONE = CN()
_CN.BACKBONE.PIXEL_FEATURE_DIM = 96

_CN.COST_REG_NET = CN()
_CN.COST_REG_NET.NUM_COST_REG_LAYER = 6
_CN.COST_REG_NET.VOLUME_REDUCTION = 'mean'
_CN.COST_REG_NET.GRID_DIM = 64
_CN.COST_REG_NET.POSITION_ENCODING = True

_CN.SDF_DECODER = CN()

_CN.POSE_INIT = CN()
_CN.POSE_INIT.PREDICT_WORLD2CAM = True
_CN.POSE_INIT.D_COARSE = 256
_CN.POSE_INIT.D_FINE = 64
_CN.POSE_INIT.FRAME_EMBED = False

_CN.POSE_INIT.BACKBONE = CN()
_CN.POSE_INIT.BACKBONE.BACKBONE_TYPE = 'ResNetFPN'
_CN.POSE_INIT.BACKBONE.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.POSE_INIT.BACKBONE.INITIAL_DIM = 128
_CN.POSE_INIT.BACKBONE.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
_CN.POSE_INIT.CROSS_ATTENTION = CN()
_CN.POSE_INIT.CROSS_ATTENTION.D_MODEL = 256
_CN.POSE_INIT.CROSS_ATTENTION.D_FFN = 256
_CN.POSE_INIT.CROSS_ATTENTION.NHEAD = 8
_CN.POSE_INIT.CROSS_ATTENTION.LAYER_NAMES = 'self-cross-self-cross-self-cross-self-cross'

_CN.POSE_INIT.CROSS_ATTENTION.ATTENTION = 'linear'  # options: ['linear', 'full']


_CN.POSE_REFINE = CN()
_CN.POSE_REFINE.USE_DOUBLE = False
_CN.POSE_REFINE.CLAMP_POSE_UPDATE = False
_CN.POSE_REFINE.LAMBDA_REG = 0.0
_CN.POSE_REFINE.LAMBDA_DAMPING = 1e2
_CN.POSE_REFINE.JOINT_STEP = 3
_CN.POSE_REFINE.POSE_UPDATE_INNER_ITER = 5
_CN.POSE_REFINE.DROP_FOR_INCREASING_ENERGY = False
_CN.POSE_REFINE.FLOWNET_NORMALIZE_FEATURE_MAP = False


_CN.POST_PROCESSING = CN()
_CN.POST_PROCESSING.JOINT_LR = 2.0e-4
_CN.POST_PROCESSING.OPTIMIZE_SCALE = True


def get_cfg_defaults():
    """Get a yacs _CNNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
