# coding=utf-8
import json
import os
from src.intention_mapping import IntentionMapper
from src.auto_report import AutoReport

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Runner(object):
    def __init__(self, config, request_json):
        self.config = config
        self.request = request_json
        self.intention_mapper = IntentionMapper(self.config, self.request)
        self.auto_reporter = AutoReport(self.request)

    def run(self):
        targets_set, mapped_ids_set = self.intention_mapper.mapping()
        user_level = self.auto_reporter.customer_intention(mapped_ids_set)
        car_info = self.auto_reporter.extract_car_info()


if __name__ == '__main__':
    with open('data/config.json', "r") as fr:
        tmp_config = json.load(fr)
    with open('data/request.json', "r") as fr:
        tmp_request = json.load(fr)
    tmp_runner = Runner(tmp_config, tmp_request)
    tmp_runner.run()
