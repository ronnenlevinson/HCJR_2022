import glob
import os

REGENERATE_GLOBALS = False

# Redefine TOP_DIR_GLOBAL as path to this shared folder on your machine.
TOP_DIR_GLOBAL = 'J:/My Drive/Hot, Cold, or Just Right Project/Task 4 (Integration & Demonstration)'
CBE_LAPTOP_DIR_GLOBAL = os.path.join(TOP_DIR_GLOBAL, 'CBE Laptop Image + Data Files')
PTR_CONTACT_TEMPERATURE_DIR_GLOBAL = os.path.join(CBE_LAPTOP_DIR_GLOBAL, 'PTR Contact Temperatures')
TRIAL_IMAGE_DIRS_VIS_GLOBAL = sorted(glob.glob(os.path.join(CBE_LAPTOP_DIR_GLOBAL, '*', 'Visual_Images')))
THERMAL_SENSATION_VOTE_DIR_GLOBAL = os.path.join(TOP_DIR_GLOBAL, 'Thermal sensation votes')
POSTPROCESSING_DIR_GLOBAL = os.path.join(TOP_DIR_GLOBAL, "Ronnen's Image Postprocessing")
VIDEO_DIR_GLOBAL = os.path.join(POSTPROCESSING_DIR_GLOBAL, 'Image videos and tables')
EXTRA_SCENE_DATA_DIR_GLOBAL = os.path.join(POSTPROCESSING_DIR_GLOBAL, 'Extra data about scenes')

# Factors used to scale color (visible) and thermal (IR) images. Color images received for
# processing by this code are 720 p high while the thermal images are 120p high.
VIS_HEIGHT_GLOBAL = 720  # pixels
IR_HEIGHT_GLOBAL = 120  # pixels
VIS_SCALE_GLOBAL = 1
IR_SCALE_GLOBAL = VIS_HEIGHT_GLOBAL / IR_HEIGHT_GLOBAL

# Space and escape characters.
SPACE_GLOBAL = chr(32)
ESC_GLOBAL = chr(27)

# Setpoint of active temperature reference (ATR) used in summer-2021 trials in the CBE climate chamber
ATR_TEMPERATURE_C = 35  # °C

# Datetime format used in image and temperature raster folder names and file names.
FILENAME_DATETIME_FORMAT_GLOBAL = "%Y-%m-%d_%H-%M-%S"

