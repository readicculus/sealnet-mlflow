import json
import os

import boto3
import cv2
import mlflow
from mlflow.entities import RunStatus

from flow import experiment, s3_cache, s3_dataset
from flow.util.s3_util import parse_s3_path, s3_check_exists


def generate(s3_client, chips_dict_uri, artifact_path):
    bucket, key = parse_s3_path(chips_dict_uri)
    exists = s3_check_exists(s3_client, bucket, key)
    if not exists:
        mlflow.set_tag("EXCEPTION", "Given labels dict uri not found: %s" % chips_dict_uri)
        mlflow.end_run(status=RunStatus.to_string(RunStatus.KILLED))
        return

    jsons = s3_cache.read_file(s3_client, bucket, key)
    chips_dict = json.loads(jsons)
    size = 0
    chip_ct = 0
    mlflow.log_metric('total_%s' % artifact_path, len(chips_dict))
    for idx,im_id in enumerate(chips_dict):
        image = chips_dict[im_id]['image']
        chips = chips_dict[im_id]['chips']
        ocv_im = cv2.imread(image['file_path'], cv2.IMREAD_UNCHANGED)
        if ocv_im is None:
            print('Error loading %s' % image['file_path'])
            continue
        for chip_meta in chips:
            chip_ct+=1
            x1 = chip_meta['chip']['x1']
            x2 = chip_meta['chip']['x2']
            y1 = chip_meta['chip']['y1']
            y2 = chip_meta['chip']['y2']
            ocv_chip = ocv_im[y1:y2, x1:x2]
            chip_fn = '%s_%d.%d_%d.%d.png' % (im_id, x1,y1,x2,y2)
            label_fn = '%s_%d.%d_%d.%d.json' % (im_id, x1,y1,x2,y2)
            dataset_uri = s3_dataset.get_dataset_uri(artifact_path)
            chip_uri = os.path.join(dataset_uri, chip_fn)
            label_uri = os.path.join(dataset_uri, label_fn)
            local_file = s3_dataset.save_image_local(ocv_chip, chip_uri) # TODO
            size += os.stat(local_file).st_size/1000000
            labels = chip_meta['labels']
            local_labels = []
            for label in labels:
                label['x1'] -= x1
                label['x2'] -= x1
                label['y1'] -= y1
                label['y2'] -= y1
                local_labels.append(label)
            local_file = s3_cache.save_json_local(local_labels, label_uri)

        mlflow.log_metric('size_Mb_%s' % artifact_path, size)
        mlflow.log_metric('processed_%s' % artifact_path, idx)
        mlflow.log_metric('chips_%s' % artifact_path, chip_ct)



def generate_chips(test_chips_uri, train_chips_uri):
    noaa_sess = boto3.session.Session(profile_name='default')
    s3_client = noaa_sess.client('s3')
    with  mlflow.start_run(run_name='generate_chips', experiment_id=experiment.experiment_id):
        mlflow.log_param('test_chips_uri', test_chips_uri)
        mlflow.log_param('train_chips_uri', train_chips_uri)
        mlflow.log_param('dataset_uri', s3_dataset.get_dataset_uri(""))

        # generate and upload test set
        generate(s3_client, test_chips_uri, 'test')
        s3_dataset.upload_external_artifacts(s3_client, 'test')

        # generate and upload train set
        generate(s3_client, train_chips_uri, 'train')
        s3_dataset.upload_external_artifacts(s3_client, 'train')


# generate_chips('s3://yboss/mlflow/1/4d395ce4121f4752892c1587674ad7c1/artifacts/test_chips.json',
#                's3://yboss/mlflow/1/4d395ce4121f4752892c1587674ad7c1/artifacts/train_chips.json')

generate_chips('s3://yboss/mlflow/3/dbcdd6194db74d62bbc32bfe87598339/artifacts/test_chips.json',
               's3://yboss/mlflow/3/dbcdd6194db74d62bbc32bfe87598339/artifacts/train_chips.json')