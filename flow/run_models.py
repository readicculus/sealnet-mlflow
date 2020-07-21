from abc import abstractmethod, ABCMeta


class MethodNotImplemented(Exception):
    pass

class Run(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def save_cam_flight_survey(self):
        pass

class ExtractRun(Run):
    def __init__(self, experiment):
        self.artifact_dir_in = 'inputs'
        self.artifact_dir_out = 'outputs'
        self.artifact_filters = 'filters.json'
        self.artifact_cfs = 'image_labels.json'
        self.experiment = experiment

    def run(self, filter):
        pass