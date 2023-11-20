python
......
class ObjectDetection:
    def __init__(self, log_name):
        self.logger = self.setup_log(log_name)
        self.detector = Detector()
        self.full_down_count = 0
        self.empty_down_count = 0
        self.full_up_count = 0
        self.empty_up_count = 0
        self.list_overlapping_blue_polygon = []
        self.list_overlapping_yellow_polygon = []

......
