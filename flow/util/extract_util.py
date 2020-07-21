import json
import os
from urllib.parse import urlparse

import cv2
import mlflow
from botocore.exceptions import ClientError
from noaadb import Session
from noaadb.schema.models import Camera, Flight, Survey, Homography


# == common
def id_from_image_name(image_name):
    return '_'.join((os.path.splitext(image_name)[0]).split('_')[:-1])

def get_json_artifact(artifact_uri, artifact_dir, artifact_name):
    uri = os.path.join(artifact_uri, artifact_dir, artifact_name)
    p = urlparse(uri)
    with open(p.path) as f:
        obj = json.load(f)
    return obj

# == Query functions ==
def query_cfs(s, cams, flights, surveys):
    qb = s.query(Camera)
    if len(cams) > 0:
        qb = qb.filter(Camera.cam_name.in_(cams))
    qb = qb.join(Flight)
    if len(flights) > 0:
        qb = qb.filter(Flight.flight_name.in_(flights))
    qb = qb.join(Survey)
    if len(surveys) > 0:  # retrieve all surveys from db
        qb = qb.filter(Survey.name.in_(surveys))
    results = qb.all()
    return results

# given a filter return all the Camera-Flight-Survey sets that the generated set of data will come from
def query_filter_cfs(s, filter):
    survey_names = filter['surveys']
    flight_names = filter['flights']
    camera_names = filter['cameras']
    results = query_cfs(s, camera_names, flight_names, survey_names)
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

def save_image_artifact(im, artifact_name, artifact_path):
    filename = '/tmp/%s' % artifact_name
    try:
        cv2.imwrite(filename, im)
        mlflow.log_artifact(filename, artifact_path)
    finally:
        os.remove(filename)

def save_list_artifact(list, artifact_name, artifact_path):
    filename = '/tmp/%s' % artifact_name
    try:
        with open(filename, 'w') as f:
            for item in list:
                f.write("%s\n" % item)
        mlflow.log_artifact(filename, artifact_path)
    finally:
        os.remove(filename)

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


# == artifacts to query objects
def artifact_to_cfs(s, json_obj):
    cams = set()
    flights = set()
    surveys = set()
    for survey in json_obj:
        surveys.add(survey)
        for flight in json_obj[survey]:
            flights.add(flight)
            for cam in json_obj[survey][flight]:
                cams.add(cam)
    results = query_cfs(s, cams, flights, surveys)
    return results



