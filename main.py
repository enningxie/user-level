# coding=utf-8
import json
import os
from src.intention_mapping import IntentionMapper
from src.auto_report import AutoReport

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.intention_mapper = IntentionMapper(self.config)
        self.auto_reporter = AutoReport()

    def run(self, request):
        response_json = {"msg": "success", "success": int(0)}
        try:
            targets_set, mapped_ids_set = self.intention_mapper.mapping(request)
            user_level = self.auto_reporter.customer_intention(mapped_ids_set, request)
            car_info = self.auto_reporter.extract_car_info()
            response_json["success"] = int(1)
            response_json['targets'] = list(targets_set)
            response_json['user_level'] = user_level
            response_json['car_info'] = car_info
        except (IOError, ImportError, ValueError, KeyError) as e:
            response_json["msg"] = str(e)
        return response_json


if __name__ == '__main__':
    with open('data/schema_01.json', "r") as fr:
        tmp_config = json.load(fr)
    with open('data/request.json', "r") as fr:
        tmp_request = json.load(fr)
    tmp_runner = Runner(tmp_config)
    response = tmp_runner.run(tmp_request)
    print(response)

    # todo, enningxie 接口封装
    # def process(self, data):
    #     """ process the request data
    #     """
    #     response_dict = {"msg": "success", "success": int(0)}
    #     try:
    #         user_response, tmp_sents, tmp_sents_label, tmp_keywords, tmp_keywords_label = self.pre_proccess(data)
    #
    #         response_dict = self.intent_detect(user_response, tmp_sents, tmp_sents_label, tmp_keywords,
    #                                            tmp_keywords_label,
    #                                            response_dict)
    #         response_dict["success"] = int(1)
    #     except (IOError, ImportError, ValueError, KeyError) as e:
    #         response_dict["msg"] = str(e)
    #     return response_dict, 0
    #
    #
    # def initialize(self):
    #     """ load module, executed once at the start of the service
    #      do service intialization and load models in this function.
    #     """
    #     self.graph_1 = tf.Graph()
    #     self.graph_2 = tf.Graph()
    #     self.sess1 = tf.Session(graph=self.graph_1)
    #     self.sess2 = tf.Session(graph=self.graph_2)
    #     with self.graph_1.as_default():
    #         with self.sess1.as_default():
    #             self.model = Bert()
    #     self.graph = tf.get_default_graph()
    #     esim
