import json
import requests
from src.Ner_tag import Ner_tag


class AutoReport:
    def __init__(self, request):
        dict_alpha = json.load(open('customer_intention.json', 'r'))
        self.label_1 = set(dict_alpha['label_1'])
        self.label_2 = set(dict_alpha['label_2'])
        self.label_3 = set(dict_alpha['label_3'])
        self.label_4 = set(dict_alpha['label_4'])
        self.interactiveTimes_alpha = dict_alpha['interactiveTimes_alpha']
        self.dialogDuration_alpha = dict_alpha['dialogDuration_alpha']
        self.callMark_alpha = dict_alpha['callMark_alpha']
        self.label_num_4_D = dict_alpha['label_num_4_D']
        self.label_num_1_A = dict_alpha['label_num_1_A']
        self.label_num_2_A = dict_alpha['label_num_2_A']
        self.label_num_2_B_C = dict_alpha['label_num_2_B_C']
        self.label_num_3_B = dict_alpha['label_num_3_B']
        # self.url = dict_alpha['url']
        self.preprocess_request(request)
        self.ner_obj = Ner_tag('data/kg.json')

    def preprocess_request(self, request):
        dialogue_data = request['dialogueData']
        self.interactive_times = dialogue_data['interactiveTimes']
        self.call_mark = dialogue_data['callMark']
        self.dialog_duration = request['communicationData']['dialogDuration']
        dialogue_text = dialogue_data['dialogueText']
        self.user_talk = [x['talkText'] for x in dialogue_text if x['talkingType'] == 2]

    def customer_intention(self, label):
        if self.call_mark != self.callMark_alpha:
            return 'E'
        else:
            label_num_1 = len(set(label) & self.label_1)
            label_num_2 = len(set(label) & self.label_2) + (
                1 if self.interactive_times >= self.interactiveTimes_alpha else 0) + (
                              1 if self.dialog_duration >= self.dialogDuration_alpha else 0)
            label_num_3 = len(set(label) & self.label_3)
            label_num_4 = len(set(label) & self.label_4)
            if label_num_4 > self.label_num_4_D:
                return 'D'
            elif label_num_1 > self.label_num_1_A:
                return 'A'
            elif label_num_2 >= self.label_num_2_A:
                return 'A'
            elif self.label_num_2_B_C[0] <= label_num_2 <= self.label_num_2_B_C[1]:
                if label_num_3 <= self.label_num_3_B:
                    return 'B'
                else:
                    return 'C'
            else:
                return 'C'

    def extract_car_info(self):
        # {'serie': ['宝马x3'], 'color': ['红色'], 'model': ['4座', '2.0t', '豪华版']}
        car_info = self.ner_obj.ner_log_api(';'.join(self.user_talk))
        return car_info

    # def aiwork(self, msg):
    #     res = requests.post(self.url, json=msg)
    #     return json.loads(res.content)
