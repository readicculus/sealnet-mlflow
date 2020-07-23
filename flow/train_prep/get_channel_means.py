from math import sqrt

import cv2
import mlflow
import numpy as np
from PyTorchYOLOv4.utils.DNData import DNData
from flow import experiment

class RollingStatistic(object):
    def __init__(self, window_size, average, variance):
        self.N = window_size
        self.average = average
        self.variance = variance
        self.stddev = sqrt(variance)

    def update(self, new, old):
        oldavg = self.average
        newavg = oldavg + (new - old)/self.N
        self.average = newavg
        self.variance += (new-old)*(new-newavg+old-oldavg)/(self.N-1)
        self.stddev = sqrt(self.variance)

dn_data = DNData()
dn_data.load('/data/s3_cache/yboss/datasets/1/yolo.data')
tr = dn_data.get_train_images()

m_bgra = np.array([0,0,0,0])
v_bgra = np.array([0,0,0,0])
with  mlflow.start_run(run_name='prepare_yolo_files', experiment_id=experiment.experiment_id):
    for i, fp in enumerate(tr):
        print('%d/%d'%(i, len(tr)))
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        b,g,r,a = cv2.split(im)
        b = np.mean(b)
        g = np.mean(g)
        r = np.mean(r)
        a = np.mean(a)
        np.variance
        m_bgra = (m_bgra * i + np.array([b,g,r,a])) / (i + 1)
        b = np.var()

    print('means')
    print(m_bgra)
    print(m_bgra/255.0)
