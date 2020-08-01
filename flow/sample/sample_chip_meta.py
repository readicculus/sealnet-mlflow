import json

import boto3
import mlflow
from mlflow.entities import RunStatus

from flow import experiment, s3_cache, s3_dataset
from flow.util.extract_util import save_dict_artifact
from flow.util.s3_util import parse_s3_path, s3_check_exists
from flow.util.sample_util import sample_chips, percent_on


def process_label_set(labels_dict_uri, s3_client, dim, overlap_w, overlap_h, overlap_thresh):
    bucket, key = parse_s3_path(labels_dict_uri)
    exists = s3_check_exists(s3_client, bucket, key)
    if not exists:
        mlflow.end_run(status=RunStatus.to_string(RunStatus.KILLED))
        mlflow.set_tag("EXCEPTION", "Given labels dict uri not found: %s" % labels_dict_uri)
        return

    jsons = s3_cache.read_file(s3_client, bucket, key)
    labels_dict = json.loads(jsons)
    im_chip_label_dict = {}
    for image_id in labels_dict:
        image = labels_dict[image_id]['image']
        labels = labels_dict[image_id]['labels']
        chips = sample_chips(image, labels, dim, overlap_w, overlap_h)

        chip_label_dict = {}  # chips
        for label in labels:
            chips_for_label = []
            for chip in chips:
                p = percent_on(chip, label)
                if p is None or p < overlap_thresh:
                    continue  # not on chip or under overlap thresh

                chips_for_label.append(chip)
            if not len(chips_for_label) > 0:
                print("Label not placed")
                print(label)
            for chip in chips_for_label:
                key = '%d_%d' % (chip['i'], chip['j'])
                if not key in chip_label_dict:
                    chip_label_dict[key] = {'labels':[], 'chip': chip}
                label.pop('image_id', None)
                chip_label_dict[key]['labels'].append(label)
        # get rid of i_j key we don't need anymore
        items = [v for k,v in chip_label_dict.items()]
        res_dict = {'image': labels_dict[image_id]['image'], 'chips': items}
        im_chip_label_dict[image_id] = res_dict
    return im_chip_label_dict

def sample_chip_meta(test_labels_uri, train_labels_uri, chip_dim, chip_overlap_w, chip_overlap_h, chip_label_overlap_thresh):
    noaa_sess = boto3.session.Session(profile_name='default')
    s3_client = noaa_sess.client('s3')
    with mlflow.start_run(run_name='sample_dataset', experiment_id=experiment.experiment_id) as mlrun:
        mlflow.set_tag('big_artifacts', True)
        mlflow.log_param('test_labels_uri', test_labels_uri)
        mlflow.log_param('train_labels_uri', train_labels_uri)
        mlflow.log_param('chip_w', chip_dim)
        mlflow.log_param('chip_h', chip_dim)
        mlflow.log_param('chip_overlap_w', chip_overlap_w)
        mlflow.log_param('chip_overlap_h', chip_overlap_h)
        mlflow.log_param('chip_label_overlap_thresh', chip_label_overlap_thresh)

        chip_labels_test = process_label_set(test_labels_uri, s3_client,
                                             chip_dim, chip_overlap_w, chip_overlap_h, chip_label_overlap_thresh)
        chip_labels_train = process_label_set(train_labels_uri, s3_client,
                                              chip_dim, chip_overlap_w, chip_overlap_h, chip_label_overlap_thresh)

        # s3_dataset.log_artifact_to_dataset()
        save_dict_artifact(chip_labels_test, 'test_chips.json', '')
        save_dict_artifact(chip_labels_train, 'train_chips.json', '')


# sample_chip_meta(
#     's3://yboss/mlflow/1/cbaa821ff3ca4cf89beb3d279a60be2a/artifacts/test_labels.json',
#     's3://yboss/mlflow/1/cbaa821ff3ca4cf89beb3d279a60be2a/artifacts/train_labels.json',
#     chip_dim=832,
#     chip_overlap_w=100,
#     chip_overlap_h=100, chip_label_overlap_thresh=0.5)

sample_chip_meta(
    's3://yboss/mlflow/3/ba27a9bdb6f7484d82b379907729aa98/artifacts/test_labels.json',
    's3://yboss/mlflow/3/ba27a9bdb6f7484d82b379907729aa98/artifacts/train_labels.json',
    chip_dim=832,
    chip_overlap_w=100,
    chip_overlap_h=100, chip_label_overlap_thresh=0.5)