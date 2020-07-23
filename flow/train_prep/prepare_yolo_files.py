import os

import boto3
import mlflow
import json

from flow import experiment, s3_dataset
from flow.util.label_formats.darknet.DataFile import DataFile
from flow.util.extract_util import save_dict_artifact, save_list_artifact
from flow.util.label_formats.darknet.darknet import darknet_format


def generate_labels(dataset_path, im_w, im_h, species_map):
    dataset_dir = s3_dataset.get_dataset_local_path(dataset_path)
    json_files = [x for x in os.listdir(dataset_dir) if x.endswith("json")]

    new_artifacts = []
    for fn in json_files:
        json_local_path = os.path.join(dataset_dir, fn)
        label_file_path = darknet_format(json_local_path, im_w, im_h, species_map)
        rk =s3_dataset.relative_key(label_file_path)
        new_artifacts.append(rk)

    return new_artifacts

def save_to_s3(s3_client, labels):
    uris = []
    for key in labels:  # upload all labels to s3
        s3_uri = s3_dataset.upload_artifact(s3_client, key)
        uris.append(s3_uri)
    return uris

def create_train_test_txt():
    # find all image paths in train and test and save to train.txt and test.txt for darknet
    test_dir = s3_dataset.get_dataset_local_path('test')
    test_image_files = [os.path.join(test_dir,x) for x in os.listdir(test_dir) if
                        x.endswith("png") or x.endswith("jpg") or x.endswith("tif")]
    train_dir = s3_dataset.get_dataset_local_path('train')
    train_image_files = [os.path.join(train_dir,x) for x in os.listdir(train_dir) if
                         x.endswith("png") or x.endswith("jpg") or x.endswith("tif")]

    # write to train.txt and test.txt
    dataset_dir = s3_dataset.get_dataset_local_path('')
    test_local_file = os.path.join(dataset_dir, 'test.txt')
    with open(test_local_file, "w") as f:
        f.write('\n'.join(test_image_files))

    train_local_file = os.path.join(dataset_dir, 'train.txt')
    with open(train_local_file, "w") as f:
        f.write('\n'.join(train_image_files))

    return test_local_file, train_local_file, test_image_files, train_image_files

def create_names_file(species_map):
    species_sorted = [None] * len(species_map)
    for key in species_map:
        idx = species_map[key]
        species_sorted[idx] = key

    dataset_dir = s3_dataset.get_dataset_local_path('')
    with open(os.path.join(dataset_dir, 'names.txt'), "w") as f:
        for el in species_sorted:
            f.write(el + '\n')
    return os.path.join(dataset_dir, 'names.txt')

def upload_yolo_files_to_s3(s3_client):
    s3_data_path = s3_dataset.upload_artifact(s3_client, 'yolo.data', checkFirst=False)
    s3_train_path = s3_dataset.upload_artifact(s3_client, 'train.txt', checkFirst=False)
    s3_test_path = s3_dataset.upload_artifact(s3_client, 'test.txt', checkFirst=False)
    mlflow.set_tag('yolo_data_file', s3_data_path)
    mlflow.set_tag('yolo_train_txt', s3_train_path)
    mlflow.set_tag('yolo_test_txt', s3_test_path)

def prepare_yolo_files(s3save = False):
    noaa_sess = boto3.session.Session(profile_name='default')
    s3_client = noaa_sess.client('s3')

    with  mlflow.start_run(run_name='prepare_yolo_files', experiment_id=experiment.experiment_id):
        mlflow.log_param('dataset_uri', s3_dataset.get_dataset_uri(""))

        with open('../util/label_formats/species_map.json') as f:
            species_map = json.load(f)
        save_dict_artifact(species_map, 'species_map.json', '')


        # generate test labels
        test_labels = generate_labels('test',832,832, species_map)
        mlflow.log_metric('test_label_files_prepared', len(test_labels))

        # generate train labels
        train_labels = generate_labels('train',832,832, species_map)
        mlflow.log_metric('train_label_files_prepared', len(train_labels))

        # create train.txt and test.txt
        test_local_path, train_local_path, test_image_files, train_image_files = create_train_test_txt()
        mlflow.log_metric('train.txt_count', len(train_image_files))
        mlflow.log_metric('test.txt_count', len(test_image_files))

        # create names.txt
        names_local_path = create_names_file(species_map)

        # create data_file
        data_file = DataFile(n_classes=len(species_map), f_names=names_local_path,
                             f_test=test_local_path, f_train=train_local_path)
        data_file_local_path = s3_dataset.get_dataset_local_path('yolo.data')
        data_file.save(data_file_local_path)

        # upload data and train/test files to s3
        upload_yolo_files_to_s3(s3_client)

        if s3save:
            # save to s3, this takes a while
            test_s3_uris = save_to_s3(s3_client, test_labels)
            save_list_artifact(test_s3_uris, 'test_label_files.txt', '')

            train_s3_uris = save_to_s3(s3_client, train_labels)
            save_list_artifact(train_s3_uris, 'train_label_files.txt', '')



if __name__ == '__main__':
    prepare_yolo_files()