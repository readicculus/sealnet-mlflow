import json

import boto3
import mlflow
from mlflow.entities import RunStatus

from flow import experiment, s3_cache
from flow.util.util import parse_s3_path, s3_check_exists, save_image_artifact
from flow.sample.util import sample_chips, plot_chips


def s3_client(args):
    pass


def prototype_chips(labels_dict_uri, chip_dim, chip_overlap_w, chip_overlap_h, image_id=None):
    noaa_sess = boto3.session.Session(profile_name='default')
    s3_client = noaa_sess.client('s3')
    with mlflow.start_run(run_name='prototype_chips', experiment_id=experiment.experiment_id) as mlrun:
        bucket, key = parse_s3_path(labels_dict_uri)
        exists = s3_check_exists(s3_client, bucket, key)
        if not exists:
            mlflow.end_run(status=RunStatus.to_string(RunStatus.KILLED))
            mlflow.set_tag("EXCEPTION", "Given labels dict uri not found: %s" % labels_dict_uri)
            return

        jsons = s3_cache.read_file(s3_client, bucket, key)
        labels_dict = json.loads(jsons)
        if image_id is None:
            image_id=list(labels_dict.keys())[0]

        image = labels_dict[image_id]['image']
        labels = labels_dict[image_id]['labels']

        mlflow.log_param('image_width', image['w'])
        mlflow.log_param('image_height', image['h'])
        mlflow.log_param('chip_w', chip_dim)
        mlflow.log_param('chip_h', chip_dim)
        mlflow.log_param('chip_overlap_w', chip_overlap_w)
        mlflow.log_param('chip_overlap_h', chip_overlap_h)

        tiles = sample_chips(image,labels, chip_dim=chip_dim, chip_overlap_w=chip_overlap_w, chip_overlap_h=chip_overlap_h)
        plot = plot_chips(image, tiles)
        save_image_artifact(plot, 'tiles.jpg', '')

        max_i = 0
        max_j = 0
        for t in tiles:
            i, j = t['i'], t['j']
            if i > max_i: max_i = i
            if j > max_j: max_j = j
        mlflow.log_metric('num_tiles', len(tiles))
        mlflow.log_metric('tiles_across', max_i+1)
        mlflow.log_metric('tiles_down', max_j+1)

prototype_chips(
    's3://yboss/mlflow/1/0c1d86c7a640459bbdff914509604b1d/artifacts/labels.json',
    chip_dim=832,
    chip_overlap_w=100,
    chip_overlap_h=100)
