import mlflow
import click
import json

# get the cam-flight-surveys applicable to given filter
from noaadb import Session
from noaadb.schema.models import EOImage, HeaderMeta, EOLabelEntry, IRLabelEntry, IRImage, EOIRLabelPair

from flow import EXPORT_METADATA_SOURCE
from flow.extract import RUN_NAME, INPUT_ARTIFACT_DIR, OUTPUT_ARTIFACT_DIR, OUTPUT_ARTIFACT_NAME
from flow.extract.util import cfs_to_dict, save_dict_artifact, query_filter_cfs, eoir_pairs_to_dict

def query_eo_ir_pairs(s, cfs_entries):
    cam_ids = [cam.id for cam in cfs_entries]
    # only joins if both eo and ir exist
    label_pairs = s.query(EOIRLabelPair)\
        .join(EOIRLabelPair.eo_label)\
        .join(EOIRLabelPair.ir_label)\
        .join(EOImage)\
        .join(HeaderMeta)\
        .filter(HeaderMeta.camera_id.in_(cam_ids))\
        .all()
    return label_pairs

def labels_by_species_counter(eoir_pair_dict):
    sp_ct = {}
    for id in eoir_pair_dict:
        image_pair = eoir_pair_dict[id]
        for label_pair in image_pair['label_pairs']:
            sp = label_pair['species']
            if sp not in sp_ct:
                sp_ct[sp] = 0
            sp_ct[sp]+=1
    return sp_ct

@click.command(help="Generate a dataset from noaadb by filtering out a subset for the experiment.")
@click.option("--filter_path", help=".")
def eo_ir_pairs_from_noaadb(filter_path):
    with mlflow.start_run(run_name=RUN_NAME) as mlrun:
        mlflow.set_tag('method_fusion', True)
        mlflow.set_tag('data-source', 'noaadb')
        mlflow.set_tag(EXPORT_METADATA_SOURCE, False)

        filters = None
        with open(filter_path) as f:
            filters = json.load(f)
        mlflow.log_artifact(filter_path, INPUT_ARTIFACT_DIR)

        s=Session()
        # get the cam-flight-surveys applicable to given filter
        cfs_entries = query_filter_cfs(s, filters)
        cfs_afct = cfs_to_dict(cfs_entries)
        save_dict_artifact(cfs_afct, 'cam_flight_surveys', INPUT_ARTIFACT_DIR)

        # generate the label metadata json file
        eoir_pairs = query_eo_ir_pairs(s, cfs_entries)
        eoir_pair_dict = eoir_pairs_to_dict(eoir_pairs)
        save_dict_artifact(eoir_pair_dict, OUTPUT_ARTIFACT_NAME, OUTPUT_ARTIFACT_DIR)

        # log meterics
        species_counts = labels_by_species_counter(eoir_pair_dict)
        for species, count in species_counts.items():
            mlflow.log_metric('species_ct_%s'%species, count)
        mlflow.log_metric('image_count', len(eoir_pair_dict))
        mlflow.log_metric('label_count', len(eoir_pairs))
        mlflow.set_tag(EXPORT_METADATA_SOURCE, True)

        s.close()

if __name__ == '__main__':
    eo_ir_pairs_from_noaadb()
