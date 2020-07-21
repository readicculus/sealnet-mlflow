from mlflow import get_experiment_by_name

from S3Cache.S3Cache import S3Cache
from S3Cache.S3Dataset import S3Dataset


experiment_name = 'fused_detector'
experiment = get_experiment_by_name(experiment_name)
s3_cache = S3Cache('/data/s3_cache/')
s3_dataset = S3Dataset(s3_cache, experiment.experiment_id)