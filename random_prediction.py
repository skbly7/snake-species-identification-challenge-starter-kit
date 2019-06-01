#!/usr/bin/env python

import aicrowd_helpers
import numpy as np
import os
import glob

AICROWD_TEST_IMAGES_PATH = os.getenv('AICROWD_TEST_IMAGES_PATH', 'data/round1')
AICROWD_PREDICTIONS_OUTPUT_PATH = os.getenv('AICROWD_PREDICTIONS_OUTPUT_PATH', 'random_prediction.csv')
########################################################################
# Register Prediction Start
########################################################################
aicrowd_helpers.execution_start()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


LINES = []

with open('data/class_idx_mapping.csv') as f:
	classes = ['filename']
	for line in f.readlines()[1:]:
		class_name = line.split(",")[0]
		classes.append(class_name) 

LINES.append(','.join(classes))

images_path = AICROWD_TEST_IMAGES_PATH + '/*.jpg'
for _file_path in glob.glob(images_path):
	########################################################################
	# Register Prediction
	#
	# Note, this prediction register is not a requirement. It is used to
	# provide you feedback of how far are you in the overall evaluation.
	# In the absence of it, the evaluation will still work, but you
	# will see progress of the evaluation as 0 until it is complete
	#
	# Here you simply announce that you completed processing a set of
	# image_names
	########################################################################
	aicrowd_helpers.execution_progress({"image_names" : [_file_path]})

	probs = softmax(np.random.rand(45))
	probs = list(map(str, probs))
	LINES.append(",".join([os.path.basename(_file_path)] + probs))

fp = open(AICROWD_PREDICTIONS_OUTPUT_PATH, "w")
fp.write("\n".join(LINES))
fp.close()
########################################################################
# Register Prediction Complete
########################################################################
aicrowd_helpers.execution_success({"predictions_output_path" : AICROWD_PREDICTIONS_OUTPUT_PATH})
