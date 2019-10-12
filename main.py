# coding=utf-8
import json
import os
from src.intention_mapping import IntentionMapper

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Runner(object):
    def __init__(self, config_path):
        with open(config_path, "r") as fr:
            self.config = json.load(fr)

    def run(self):
        intention_mapper = IntentionMapper(self.config)
        return intention_mapper.mapping()


if __name__ == '__main__':
    tmp_runner = Runner('data/config.json')
    targets_set, mapped_ids_set = tmp_runner.run()
    print(targets_set)
    print(mapped_ids_set)
