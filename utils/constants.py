from __future__ import division

LOSS_MEAN_SQUARED_ERROR = "mean_squared_error"
LOSS_MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
LOSS_CUSTOM_PW_SQUARED_ERROR = "custom_pw_squared_error"

ADV_LOSS_MEAN_SQUARED_ERROR = "adv_mean_squared_error"
ADV_LOSS_ABS_SQUARED_ERROR = "adv_abs_squared_error"

DEFAULT_CHOICE_TAIL =  10
CONFIG_DEFAULT_FILE_NAME = 'config.json'

NONE = 'None'
TRUE = 'True'


ILP_PROP_BIN_INPUT_VARS = "property_"
ILP_NONLINEAR_BIN_VARS = "nonlinear_"
ILP_INPUT_VARS = "signal_"
ILP_BIN_INPUT_VARS = "bin_signal_"
ILP_STATE_VARS = "state_"
ILP_OUTPUT_VARS = "output_"
ILP_Y_VARS = "y_"
ILP_Z_VARS = "z_"
ILP_CHOICE_VARS = "choice_"
ILP_CHOICE_BOOL_VARS = "bool_choice_"
ILP_THRESH_BOOL_VARS= "bool_thresh_"

INPUT_VARS_TAG = "inputs"
OUTPUT_VARS_TAG = "outputs"
STATE_VARS_TAG = "states"
CHOICE_VAR_TAG = "choice"
