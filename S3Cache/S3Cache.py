import json
import os
import cv2
import mlflow
import torch
from mlflow import pytorch

class S3Cache():
    def __init__(self, local_dir):
        self.local_dir = local_dir

    def parse_s3_path(self, path):
        _, path = path.split(":", 1)
        path = path.lstrip("/")
        bucket, key = path.split("/", 1)
        return bucket, key

    def read_file(self, s3_client, bucket, key):
        local_path = os.path.join(self.local_dir, bucket, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            print("Downloading from S3 -> %s" % local_path)
            s3_client.download_file(bucket, key, local_path)

        with open(local_path, 'r') as file:
            data = file.read()
        return data

    def get_artifact_local_path(self, uri):
        bucket, key = self.parse_s3_path(uri)
        local_path = os.path.join(self.local_dir, bucket, key)
        return local_path

    def save_model(self,model, artifact_path, name):
        mlflow_artifact_models_uri = mlflow.get_artifact_uri(artifact_path)
        local_path = self.get_artifact_local_path(mlflow_artifact_models_uri)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mlflow.pytorch.save_model(model, os.path.join(local_path, name))

    def save_ckpt(self,ckpt, artifact_path, name):
        mlflow_artifact_models_uri = mlflow.get_artifact_uri(artifact_path)
        local_path = self.get_artifact_local_path(mlflow_artifact_models_uri)
        os.makedirs(local_path, exist_ok=True)
        torch.save(ckpt, os.path.join(local_path, name))
        return os.path.join(local_path, name)

    # given a cv2 image, cache locally and upload
    def save_image_local(self, im, uri):
        local_path = self.get_artifact_local_path(uri)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            cv2.imwrite(local_path, im)
        # if not s3_check_exists(s3_client, bucket, key):
        #     print("Uploading image to s3 -> s3://%s/%s" % (bucket,key))
        #     s3_upload_image(s3_client, im, bucket, key)
        print()
        return local_path

    def save_list_local(self, l, uri):
        local_path = self.get_artifact_local_path(uri)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            # print("Saving list locally -> %s" % local_path)
            with open(local_path, 'w') as f:
                for item in l:
                    f.write("%s\n" % item)
        return local_path

    def save_json_local(self,obj,uri):
        local_path = self.get_artifact_local_path(uri)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            # print("Saving list locally -> %s" % local_path)
            with open(local_path, 'w') as f:
                json.dump(obj, f)
        return local_path
