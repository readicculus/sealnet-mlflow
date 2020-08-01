from mlflow import get_experiment_by_name

from S3Cache.S3Cache import S3Cache
from S3Cache.S3Dataset import S3Dataset
from S3Cache.S3Models import S3Models

# experiment_name = 'eo_detector'
experiment_name = 'refine_fused_alignment'
experiment = get_experiment_by_name(experiment_name)
s3_cache = S3Cache('/fast/s3_cache/')
s3_dataset = S3Dataset(s3_cache, experiment.experiment_id)
s3_models = S3Models(s3_cache, experiment.experiment_id)

