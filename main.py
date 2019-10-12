# coding=utf-8
import json
import os
from src.intention_mapping import IntentionMapper

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Runner(object):
    def __init__(self, config):
        self.config = config

    def run(self):
        intention_mapper = IntentionMapper(self.config)
        return intention_mapper.mapping()


if __name__ == '__main__':
    with open('data/config.json', "r") as fr:
        tmp_config = json.load(fr)
    tmp_runner = Runner(tmp_config)
    targets_set, mapped_ids_set = tmp_runner.run()
    # todo user: liangyang
