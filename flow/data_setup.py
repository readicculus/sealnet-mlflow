import os

import click
import mlflow

from flow import extract, EXPORT_METADATA_SOURCE

client = mlflow.tracking.MlflowClient()

from flow.extract.eo_ir_pairs_from_noaadb import eo_ir_pairs_from_noaadb

def check_if_src_exists(src):
    client = mlflow.tracking.MlflowClient()
    all_run_infos = client.list_run_infos('0')
    runs_with_tag = []
    for run_info in all_run_infos:
        if run_info.status == 'FAILED':
            continue
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if not src in tags:
            continue
        if tags[src] == 'True':
            runs_with_tag.append(full_run)

    if len(runs_with_tag) == 0:
        return None

    assert(len(runs_with_tag) == 1) # only 1 run is allowed to be tag.src = True

    return runs_with_tag[0]

def _get_or_run(entry_point, src, params):
    submitted_run = check_if_src_exists(src)
    if not submitted_run:
        res = mlflow.run(".", entry_point, parameters=params, use_conda=False)
        submitted_run = mlflow.tracking.MlflowClient().get_run(res.run_id)
    return submitted_run

@click.command(help="Generate a dataset from noaadb by filtering out a subset for the experiment.")
@click.option("--filter_path", help=".")
def data_setup(filter_path):
    mlflow.active_run()
    export_meta_run = _get_or_run('eo_ir_pairs_from_noaadb', EXPORT_METADATA_SOURCE ,{'filter_path': filter_path})
    uri = os.path.join(export_meta_run.info.artifact_uri, extract.OUTPUT_ARTIFACT_DIR, extract.OUTPUT_ARTIFACT_NAME)
    x=1

if __name__ == '__main__':
    data_setup()