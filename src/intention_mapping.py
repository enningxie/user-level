# coding=utf-8
import json
from src.bert import BertRunner
from src.esim import ESIM
from src.esim_config import ESIMConfig
import re
import numpy as np


class IntentionMapper(object):
    def __init__(self, schema, request=None):
        self.preprocess_schema(schema)
        self.preprocess_request(request)
        self.pattern = r',|\.|;|\?|\~|!|，|。|、|；|·|！| |…|\n|\？|\～|\\n'
        self.bert = BertRunner()
        self.esim = ESIM(ESIMConfig())
        self.yes_or_no = [0, 1]

    def preprocess_schema(self, raw_schema, filter_flag=False, min_len=2):
        self.schema_id = raw_schema['SchemaId']
        self.id2name = {}
        self.utterances = []
        self.utterances_label = []
        for tmp_intent in raw_schema['Intents']:
            self.id2name[tmp_intent['IntentId']] = tmp_intent['IntentName']
            tmp_utterances = tmp_intent['Utterances']
            # 过滤短句，默认大于长度2
            if filter_flag:
                tmp_utterances = [x for x in tmp_utterances if len(x) > min_len]
            self.utterances.extend(tmp_utterances)
            self.utterances_label.extend([tmp_intent['IntentId']] * len(tmp_utterances))
        self.name2id = {tmp_value: tmp_key for tmp_key, tmp_value in self.id2name.items()}
        self.ai_id2name = {}
        self.ai_utterances = []
        self.ai_utterances_label = []
        self.ai_id2polarity = {}
        for tmp_ai_sentence in raw_schema['AiSentence']:
            self.ai_id2name[tmp_ai_sentence['AiIntentId']] = tmp_ai_sentence['AiIntentName']
            self.ai_id2polarity[tmp_ai_sentence['AiIntentId']] = [tmp_ai_sentence['PositiveIntentId'],
                                                                  tmp_ai_sentence['NegativeIntentId']]
            tmp_ai_utterances = tmp_ai_sentence['AiUtterances']
            self.ai_utterances.extend(tmp_ai_utterances)
            self.ai_utterances_label.extend([tmp_ai_sentence['AiIntentId']] * len(tmp_ai_utterances))

    # 处理原始请求json todo
    def preprocess_request(self, raw_request):
        dialogue_data = raw_request['dialogueData']
        # 对话id到AI表述的映射
        self.dialogUuid2talkText = {}
        for tmp_dialogue_text in dialogue_data['dialogueText']:
            if tmp_dialogue_text['talkingType'] == 1:
                self.dialogUuid2talkText[tmp_dialogue_text['dialogUuid']] = tmp_dialogue_text['talkText']
        self.hit_utterances = []
        self.hit_dialogUuids = []
        # self.hit_intention_ids = []
        for tmp_intent_data in dialogue_data['intentionData']:
            # 意图匹配成功
            if tmp_intent_data['matchedStatus'] == 1:
                # self.hit_intention_ids.append(tmp_intent_data['matchedIntentionId'])
                self.hit_dialogUuids.append(tmp_intent_data['dialogUuid'])
                # 关键词匹配意图成功
                # todo 字段可能不一样
                if 'keyword' in tmp_intent_data:
                    tmp_utterance = tmp_intent_data['keyword']
                else:
                    tmp_utterance = tmp_intent_data['topMatchedSentence']
                self.hit_utterances.append(tmp_utterance)

    def split_and_filter(self, source_ai_utterance, min_len=5):
        return [tmp_str.strip() for tmp_str in re.split(self.pattern, source_ai_utterance) if
                len(tmp_str.strip()) > min_len]

    def mapping(self, esim_threshold=0.5, bert_threshold=0.2):
        targets = []
        mapped_ids = []
        # esim part
        # todo 批量预测，可以大幅度加速模型推理速度
        for tmp_index, (tmp_dialogUuid, tmp_utterance) in enumerate(zip(self.hit_dialogUuids, self.hit_utterances)):
            esim_pred = self.esim.predict([tmp_utterance] * len(self.utterances), self.utterances)
            # todo bug
            most_sim_index = np.argmax(esim_pred).item()
            # esim最大相似度超过阈值
            if esim_pred[most_sim_index].item() > esim_threshold:
                mapped_class = self.utterances_label[most_sim_index]
                # 映射的class为肯定或否定, 则需要使用bert进行ai_utterance相似度度量
                if mapped_class in self.yes_or_no:
                    tmp_ai_utterance = self.dialogUuid2talkText[tmp_dialogUuid]
                    # 对当前ai表述进行切分
                    tmp_ai_utterance_split = self.split_and_filter(tmp_ai_utterance)
                    mapped_ai_class = -1
                    max_sim_score = 0.
                    for part_ai_utterance in tmp_ai_utterance_split:
                        bert_pred = self.bert.predict([part_ai_utterance] * len(self.ai_utterances), self.ai_utterances)
                        # todo bug
                        most_sim_index_ = np.argmax(bert_pred).item()
                        # bert相似度超过阈值
                        tmp_sim_score = bert_pred[most_sim_index_].item()
                        if tmp_sim_score > bert_threshold and tmp_sim_score > max_sim_score:
                            max_sim_score = tmp_sim_score
                            mapped_ai_class = self.ai_utterances_label[most_sim_index_]
                    # 有映射到具体的ai_class
                    if mapped_ai_class != -1:
                        # mapped_class 为肯定
                        if mapped_class == 0:
                            mapped_class = self.ai_id2polarity[mapped_ai_class][0]
                        else:
                            mapped_class = self.ai_id2polarity[mapped_ai_class][1]
                targets.append(self.id2name[mapped_class])
                mapped_ids.append(mapped_class)
        targets_set = set(targets)
        mapped_ids_set = set(mapped_ids)
        assert len(targets_set) == len(mapped_ids_set), 'targets and mapped_ids has different length.'
        return targets_set, mapped_ids_set


if __name__ == '__main__':
    with open('../data/schema_01.json', "r") as fr:
        tmp_schema = json.load(fr)

    tmp_intention_mapper = IntentionMapper(tmp_schema)
    print(tmp_intention_mapper.ai_utterances)
    print(tmp_intention_mapper.ai_utterances_label)
    print(tmp_intention_mapper.ai_id2polarity)
    assert len(tmp_intention_mapper.ai_utterances) == len(tmp_intention_mapper.ai_utterances_label), 'error'
