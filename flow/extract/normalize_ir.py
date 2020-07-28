import inspect

import cv2
import mlflow
import click

# get the cam-flight-surveys applicable to given filter
from noaadb import Session
from noaadb.schema.models.ml_data import TrainTestSplit, MLType
from noaadb.schema.models.survey_data import IRImage
from sqlalchemy.orm import aliased

from flow import experiment, s3_cache
import numpy as np


# normalize IR
def min_max_norm(im):
    im_ir = ((im - np.min(im)) / (0.0 + np.max(im) - np.min(im))) * 256.0
    im_ir = im_ir.astype(np.uint8)
    return im_ir

def normalize_ir():
    with mlflow.start_run(run_name='normalize_ir_images', experiment_id=experiment.experiment_id) as mlrun:
        mlflow.set_tag('method_fusion', True)
        mlflow.set_tag('data-source', 'noaadb')


        uri, local_path = s3_cache.log_function('normalization_function.txt', min_max_norm)

        s=Session()

        # get all IR images
        im_rows = s.query(IRImage).all()
        mlflow.log_metric('image_count', len(im_rows))

        for image in im_rows:
            fp = image.file_path
            im = cv2.imread(fp, cv2.IMREAD_ANYDEPTH)
        # end session
        s.close()


if __name__ == '__main__':
    normalize_ir()
