import json
import os

import cv2
from botocore.exceptions import ClientError

# parse a path of this format s3://bucket/key
def parse_s3_path(path):
    _, path = path.split(":", 1)
    path = path.lstrip("/")
    bucket, key = path.split("/", 1)
    return bucket, key

def s3_check_exists(s3_client, bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True

def s3_upload_image(s3_client, im, bucket_name, image_key):
    is_success, im_buf_arr = cv2.imencode(".png", im)
    byte_im = im_buf_arr.tobytes()
    response = s3_client.put_object(Bucket=bucket_name, Key=image_key, Body=byte_im, ContentType='image/png')
    print("Response: {}".format(response))  # See result of request.
