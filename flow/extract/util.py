import json
import os

import mlflow
from noaadb import Session
from noaadb.schema.models import Camera, Flight, Survey, Homography


# == common
def id_from_image_name(image_name):
    return '_'.join((os.path.splitext(image_name)[0]).split('_')[:-1])


# == Query functions ==
# given a filter return all the Camera-Flight-Survey sets that the generated set of data will come from
def query_filter_cfs(s, filter):
    survey_names = filter['surveys']
    flight_names = filter['flights']
    camera_names = filter['cameras']
    qb = s.query(Camera)
    if len(camera_names) > 0:
        qb = qb.filter(Camera.cam_name.in_(camera_names))
    qb = qb.join(Flight)
    if len(flight_names) > 0:
        qb = qb.filter(Flight.flight_name.in_(flight_names))
    qb = qb.join(Survey)
    if len(survey_names) > 0: # retrieve all surveys from db
        qb = qb.filter(Survey.name.in_(survey_names))
    results = qb.all()
    return results

def query_homographies(s, cfs_entries):
    cam_ids = [cam.id for cam in cfs_entries]
    homogs = s.query(Homography).filter(Homography.camera_id.in_(cam_ids)).all()
    return homogs

# == Artifact generators ==
def save_dict_artifact(obj, artifact_name, artifact_path):
    filename = '/tmp/%s' % artifact_name
    temp = open(filename, 'w')
    try:
        json.dump(obj, temp)
        temp.flush()  # make sure all data is flushed to disk
        mlflow.log_artifact(filename, artifact_path)
    finally:
        temp.close()
        os.remove(filename)

def eoir_pairs_to_dict(eoir_pairs):
    pair_dict = {}
    for i, pair in enumerate(eoir_pairs):
        d = pair.to_dict()
        im_id = id_from_image_name(d['eo_image'])
        im_id_ir = id_from_image_name(d['ir_image'])
        assert(im_id == im_id_ir)
        if not im_id in pair_dict:
            pair_dict[im_id] = {'eo_image': d['eo_image'],
                                'ir_image': d['ir_image'],
                                'label_pairs': []}

        del d['eo_image']
        del d['ir_image']
        pair_dict[im_id]['label_pairs'].append(d)
    return pair_dict

def cfs_to_dict(cams):
    cfs_dict = {}
    for cam in cams:
        cam_name = cam.cam_name
        flight_name = cam.flight.flight_name
        survey_name = cam.flight.survey.name
        if not survey_name in cfs_dict:
            cfs_dict[survey_name] = {}
        if not flight_name in cfs_dict[survey_name]:
            cfs_dict[survey_name][flight_name] = []
        cfs_dict[survey_name][flight_name].append(cam_name)
    return cfs_dict


