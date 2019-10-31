# -*- coding: utf-8 -*-
import allspark
from src.intention_mapping import IntentionMapper
from src.auto_report import AutoReport
import json


class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is a example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """

    def initialize(self):
        """ load module, executed once at the start of the service
         do service intialization and load models in this function.
        """
        with open('data/schema_01.json', "r") as fr:
            tmp_config = json.load(fr)
        self.config = tmp_config
        self.intention_mapper = IntentionMapper(self.config)
        self.auto_reporter = AutoReport()

    def process(self, data):
        """ process the request data
        """
        request = json.loads(str(data, encoding="utf-8"))
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
        return response_json, 0


if __name__ == '__main__':
    # paramter worker_threads indicates concurrency of processing
    runner = MyProcessor(worker_threads=10)
    runner.run()
