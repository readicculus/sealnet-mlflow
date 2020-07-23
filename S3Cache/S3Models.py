import os

import mlflow


class S3Models():
    def __init__(self, s3_cache, experiment_id, bucket = "yboss"):
        self.s3_cache = s3_cache
        self.bucket = bucket
        self.root_directory = "models"
        self.experiment_id = experiment_id
        self.local_dataset_base = os.path.join(self.s3_cache.local_dir, self.bucket, self.root_directory, experiment_id)
        self.remote_dataset_base = 's3://' + os.path.join(self.bucket, self.root_directory, experiment_id)
        os.makedirs(self.local_dataset_base, exist_ok=True)



    def save(self):
        run = mlflow.active_run()
        run_id = None if run is None else run.info.run_id
        x=1
