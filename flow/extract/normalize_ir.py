import inspect

import cv2
import mlflow
import click

# get the cam-flight-surveys applicable to given filter
from noaadb import Session
from noaadb.schema.models.ml_data import TrainTestSplit, MLType
from noaadb.schema.models.survey_data import IRImage
from sqlalchemy.orm import aliased

from flow import experiment, s3_cache, s3_dataset
import numpy as np


# normalize IR
def min_max_norm(im):
    im_ir = ((im - np.min(im)) / (0.0 + np.max(im) - np.min(im)))
    im_ir = im_ir*255.0
    im_ir = im_ir.astype(np.uint8)
    print(im_ir.min(), im_ir.max())
    return im_ir

def normalize_ir():
    with mlflow.start_run(run_name='normalize_ir_images', experiment_id=experiment.experiment_id) as mlrun:
        mlflow.set_tag('method_fusion', True)
        mlflow.set_tag('data-source', 'noaadb')
        mlflow.set_tag('big_artifacts', True)

        # log the function used for normalization
        _, _ = s3_cache.log_function('normalization_function.txt', min_max_norm)


        mlflow.log_param('data_uri', s3_dataset.get_dataset_uri('normalized_ir'))
        mlflow.log_param('data_path', s3_dataset.get_dataset_local_path('normalized_ir'))

        s=Session()

        # get all IR images
        im_rows = s.query(IRImage).all()
        mlflow.log_metric('image_count', len(im_rows))
        progress = 0
        issue_image_list = []
        for image in im_rows:
            fp = image.file_path
            im = cv2.imread(fp, cv2.IMREAD_ANYDEPTH)
            if im is None:
                issue_image_list.append(image.file_name)
                continue
            im_norm = min_max_norm(im)
            file_uri = s3_dataset.get_dataset_uri('normalized_ir/%s' % image.file_name)
            s3_dataset.save_image_local(im_norm, file_uri)
            progress += 1
            mlflow.log_metric('progress', progress)

        if len(issue_image_list) > 0:
            uri = mlflow.get_artifact_uri('could_not_load_images.txt')
            local_path = s3_cache.save_list_local(issue_image_list, uri)
            mlflow.log_artifact(local_path)
        # end session
        s.close()


if __name__ == '__main__':
    normalize_ir()
