import os


class S3Dataset():
    def __init__(self, s3_cache, experiment_id, bucket = "yboss"):
        self.s3_cache = s3_cache
        self.bucket = bucket
        self.root_directory = "datasets"
        self.experiment_id = experiment_id
        self.local_dataset_base = os.path.join(self.s3_cache.local_dir, self.bucket, self.root_directory, experiment_id)
        self.remote_dataset_base = 's3://' + os.path.join(self.bucket, self.root_directory, experiment_id)
        os.makedirs(self.local_dataset_base, exist_ok=True)

    def get_dataset_uri(self, dataset_path):
        return os.path.join(self.remote_dataset_base, dataset_path)

    def get_dataset_local_path(self, dataset_path):
        return os.path.join(self.local_dataset_base, dataset_path)

    def save_image_local(self, im, uri):
        return self.s3_cache.save_image_local(im, uri)

    def save_json_local(self,obj,uri):
        return self.s3_cache.save_json_local(obj,uri)

    def relative_key(self, dataset_path):
        local_path = self.get_dataset_local_path(dataset_path)
        return os.path.relpath(local_path, self.local_dataset_base)

    def upload_artifact(self, s3_client, artifact_path, checkFirst = True):
        local_file = self.get_dataset_local_path(artifact_path)
        s3_uri = self.get_dataset_uri(artifact_path)
        bucket, key = self.parse_s3_path(s3_uri)

        if checkFirst:
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
            except:
                s3_client.upload_file(local_file, bucket, key)
        else:
            s3_client.upload_file(local_file, bucket, key)

        return s3_uri

    def upload_external_artifacts(self, s3_client,  dataset_path):
        local_path = self.get_dataset_local_path(dataset_path)
        for root, dirs, files in os.walk(local_path):
            for filename in files:
                # construct the full local path
                local_file = os.path.join(root, filename)
                dataset_file_path =self.relative_key(local_file)
                s3_uri = self.get_dataset_uri(dataset_file_path)
                bucket, key = self.parse_s3_path(s3_uri)

                try:
                    s3_client.head_object(Bucket=bucket, Key=key)
                except:
                    s3_client.upload_file(local_file, bucket, key)


    def parse_s3_path(self, path):
        _, path = path.split(":", 1)
        path = path.lstrip("/")
        bucket, key = path.split("/", 1)
        return bucket, key
