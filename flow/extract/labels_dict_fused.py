import mlflow
import click
import json

# get the cam-flight-surveys applicable to given filter
from noaadb import Session
from noaadb.schema.models import EOImage, HeaderMeta, EOLabelEntry, Species
from noaadb.schema.models.ml_data import TrainTestSplit, MLType
from noaadb.schema.models.survey_data import FusedImage
from sqlalchemy.orm import aliased
from flow import experiment

from flow.util.extract_util import cfs_to_dict, save_dict_artifact, query_filter_cfs, id_from_image_name

# map a label with species_x -> one or many labels with species_a,b,c..
# used when UNK seal we can train with UNK seal as Bearded and Ringed
def map_species(species_mappings, label_dict):
    sp = label_dict['species']
    if sp in species_mappings:
        mapped_species = species_mappings[sp]
        mapped_labels = []
        for dest in mapped_species:
            dnew = dict(label_dict)
            dnew['species'] = dest
            mapped_labels.append(dnew)
        return mapped_labels
    else:
        return [label_dict]

def query_fused_labels(s, cfs_entries, species_filter, species_mappings):
    # get all relevant cam ids
    cam_ids = [cam.id for cam in cfs_entries]

    # get the fused images with headers from that camera
    eo_im = aliased(EOImage)
    fused = s.query(FusedImage, TrainTestSplit)\
        .join(TrainTestSplit, TrainTestSplit.image_id == FusedImage.eo_image_id)\
        .join(FusedImage.eo_image).join(HeaderMeta)\
        .filter(HeaderMeta.camera_id.in_(cam_ids)).all()


    # get labels for the eo image of the fused image
    eo_image_ids = [f.eo_image_id for f,tts in fused]
    labels_query = s.query(EOLabelEntry).filter(EOLabelEntry.image_id.in_(eo_image_ids))
    if len(species_filter) > 0:
        labels_query = labels_query.join(Species).filter(Species.name.in_(species_filter))
    labels = labels_query.all()

    # create our metadata of labels for each fused image
    im_dict_test = {}
    im_dict_train = {}
    for f, tts in fused:
        im_id = id_from_image_name(f.file_name)
        if tts.type == MLType.TRAIN:
            im_dict_train[im_id] = {'image': f.to_dict(), 'labels': []}
        elif tts.type == MLType.TEST:
            im_dict_test[im_id] = {'image': f.to_dict(), 'labels': []}
        else:
            raise Exception('Label neither train nor test')

    for l in labels:
        im_id = id_from_image_name(l.image_id)
        label_dict = l.to_dict()
        mapped_labels = map_species(species_mappings, label_dict)
        for mapped_label in mapped_labels:
            if im_id in im_dict_test:
                im_dict_test[im_id]['labels'].append(mapped_label)
            else:
                im_dict_train[im_id]['labels'].append(mapped_label)

    return im_dict_train, im_dict_test

def labels_by_species_counter(d):
    species_cts = {}
    label_count = 0
    for id in d:
        obj = d[id]
        image = obj['image']
        labels = obj['labels']
        for l in labels:
            label_count += 1
            if not l['species'] in species_cts:
                species_cts[l['species']] = 0
            species_cts[l['species']] += 1

    return species_cts, label_count

def fused_labels_dict(filter_path):
    with mlflow.start_run(run_name='generate_fused_labels_dict', experiment_id=experiment.experiment_id) as mlrun:
        mlflow.set_tag('method_fusion', True)
        mlflow.set_tag('data-source', 'noaadb')

        filters = None
        with open(filter_path) as f:
            filters = json.load(f)
        mlflow.log_artifact(filter_path)

        s=Session()

        # get the cam-flight-surveys applicable to given filter
        cfs_entries = query_filter_cfs(s, filters)
        cfs_artifact = cfs_to_dict(cfs_entries)
        save_dict_artifact(cfs_artifact, 'cam_flight_surveys.json', '')

        # generate the label metadata json file
        train_dict, test_dict = query_fused_labels(s, cfs_entries, filters['species'], filters['species_mappings'])

        save_dict_artifact(train_dict, 'train_labels.json', '')
        save_dict_artifact(test_dict, 'test_labels.json', '')

        # log meterics
        species_counts_test, label_count_test = labels_by_species_counter(test_dict)
        species_counts_train, label_count_train = labels_by_species_counter(train_dict)
        species_counts_total, label_count_total = labels_by_species_counter({**train_dict, **test_dict})
        for species, count in species_counts_test.items():
            mlflow.log_metric('%s_test'%species, count)
        for species, count in species_counts_train.items():
            mlflow.log_metric('%s_train'%species, count)
        for species, count in species_counts_total.items():
            mlflow.log_metric('%s_total'%species, count)
        mlflow.log_metric('images_total', len(test_dict) + len(train_dict))
        mlflow.log_metric('images_test', len(test_dict))
        mlflow.log_metric('images_train', len(train_dict))
        mlflow.log_metric('labels_total', label_count_total)
        mlflow.log_metric('labels_test', label_count_test)
        mlflow.log_metric('labels_train', label_count_train)

        # end session
        s.close()

@click.command(help="Generate a dataset from noaadb by filtering out a subset for the experiment.")
@click.option("--filter_path", help=".")
def eo_ir_pairs_from_noaadb(filter_path):
    fused_labels_dict(filter_path)

if __name__ == '__main__':
    fused_labels_dict('/home/yuval/Documents/XNOR/sealnet-mlflow/flow/extract/labels_dict_filters.json')
