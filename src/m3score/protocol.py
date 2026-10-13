"""Frozen constants of the m3-v1 protocol.

Changing any value here changes what an M3-Score means, so it requires a new
PROTOCOL_VERSION.
"""

PROTOCOL_VERSION = "m3-v1"
CUSTOM_PROTOCOL = "custom"

# Encoder
BACKBONE_ID = "Snarcy/RadioDino-s16"
BACKBONE_REVISION = "773d28d873455153495d39179fc42ea816aa1212"
IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Read-out depth of each axis, counted from 1
FIDELITY_LAYER = 12
MEMORIZATION_LAYER = 9
COVERAGE_LAYER = 4
LAYERS = (FIDELITY_LAYER, MEMORIZATION_LAYER, COVERAGE_LAYER)

# Statistics
RBF_BANDWIDTH_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
COVERAGE_K = 5
N_BOOTSTRAP = 200
N_PERMUTATIONS = 200
CI_ALPHA = 0.05
FLOOR_SPLITS = 10

# Volumes
SLICE_BAND = (55 / 155, 86 / 155)
INTENSITY_PERCENTILES = (0.5, 99.5)

# Sample sizes
MAX_IMAGES = 5000
MIN_IMAGES_WARN = 500
MIN_SUBJECTS_WARN = 50
MIN_REAL_IMAGES = COVERAGE_K + 1
MIN_GEN_IMAGES = 2
# The statistics hold a kernel over every accumulated image, so memory grows with the
# square of the total. Accumulation stops here with an explanation instead of running
# out of memory after all the features have been extracted.
MAX_TOTAL_IMAGES = 4 * MAX_IMAGES
