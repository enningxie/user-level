"""
################################Ner_tag.py################################
程序名称:     Ner_tag.py
功能描述:     实体标注
创建人名:     wuxinhui
创建日期:     2019-07-12
版本说明:     v1.0
################################Ner_tag.py################################
"""

import numpy as np
import re
from random import shuffle
import copy
import jieba
import cn2an
import json
import random


def utils_func(src):
    def strQ2B(ustring):

        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    return strQ2B(src).strip().lower()


# functions
# find the ids of str for tar
def find_str(src, tar, idl):
    ids = []
    if idl == 0:
        for i in range(0, len(src) - len(tar) + 1):
            if src[i:i + len(tar)] == tar:
                ids.append(i)
    else:
        for i in range(0, len(src) - len(tar) + 1):
            if src[i:i + len(tar)] == tar and reg_parse(src, i, i + len(tar) - 1) == True:
                ids.append(i)
    return ids


# find the parse for src
def reg_parse(src, i, j):
    R1, R2 = False, False
    if i == 0:
        R1 = True
    else:
        if re.match("[a-z0-9\\-]{2}", src[i - 1:i + 1], re.I) == None:
            R1 = True
    if j == len(src) - 1:
        R2 = True
    else:
        if re.match("[a-z0-9\\-]{2}", src[j:j + 2], re.I) == None:
            R2 = True
    if R1 == True and R2 == True:
        return True
    else:
        return False


# the tag functions
def tag_func(sen, S, tag, label, Allow, idl):
    S = set(S)
    S = [s for s in S if s in sen]
    # 提取索引列表
    idList = []
    for i in S:
        ids = find_str(sen, i, idl)
        ids = [list(range(w, w + len(i))) for w in ids]
        idList.extend(ids)
    idList.sort(key=len)
    idList.reverse()

    """
    # 去重索引列表
    idSet = []
    idList.sort(key=len)
    while(len(idList) != 0):
        temp = idList.pop()
        lab = 0
        for i in idSet:
            if len(set(temp).intersection(set(i))) > 0:
                lab = 1
                break
        if lab == 0:
            idSet.append(temp)
    """

    # 标注索引列表
    for ids in idList:
        table = [tag[i][0] for i in ids]
        flag = [tag[ids[0]], tag[ids[-1]]]
        if not (set(table).issubset(set(Allow))):
            continue
        if re.search("O$|BEG$", flag[0], re.I) == None or re.search("O$|END$", flag[1], re.I) == None:
            continue
        if len(ids) > 1:
            tag[ids[0]] = label + "_BEG"
            tag[ids[-1]] = label + "_END"
            for i in ids[1:-1]:
                tag[i] = label + "_MID"
        else:
            tag[ids[0]] = label
    return tag


# extract the tag from sen
def tag_extract(sen, tag, label):
    labelL = []
    for ids in range(len(sen)):
        if tag[ids] == label + "_BEG":
            tmp = sen[ids]
        elif tag[ids] == label + "_MID":
            tmp += sen[ids]
        elif tag[ids] == label + "_END":
            tmp += sen[ids]
            labelL.append(tmp)
        elif tag[ids] == label:
            labelL.append(sen[ids])
        else:
            tmp = ""
    return labelL


# classes
# main Ner_tag spi class
class Ner_tag(object):
    """docstring for Ner_tag"""

    def __init__(self, file):
        super(Ner_tag, self).__init__()

        self.__kg = json.load(open(file, "rb"))

    def set_ner_kg(self, kg):

        self.__kg = kg
        return

    def get_ner_kg(self):

        return self.__kg

    def ner_tag_api(self, sen):
        """
        finsh the tag of the sentence, acquire all tags | char level
        """
        sen = utils_func(sen)
        tag = ["O"] * len(sen)
        labels = [l for l in self.__kg.keys() if l not in ["B", "S", "M"]]

        for l in labels:
            try:
                regexp = [re.compile(r) for r in self.__kg[l]["regexp"]]
                S = sum([r.findall(sen) for r in regexp], [])
            except:
                value = self.__kg[l]["value"]
                S = [v for v in value if v in sen]
            tag = tag_func(sen, S, tag, l, ["O"], 0)

        tag = tag_func(sen, self.__kg["B"]["value"], tag, "B", ["O"], 1)
        tag = tag_func(sen, self.__kg["S"]["value"], tag, "S", ["O"], 0)
        tag = tag_func(sen, self.__kg["M"]["value"], tag, "M", ["S", "O"], 0)

        Blabel = tag_extract(sen, tag, "B")
        Stalk = sum([self.__kg["S"]["map"][b] for b in Blabel], [])
        tag = tag_func(sen, Stalk, tag, "S", ["O"], 1)
        Mtalk = sum([self.__kg["M"]["map"][b] for b in Blabel], [])
        tag = tag_func(sen, Mtalk, tag, "M", ["O"], 1)

        return tag

    def ner_log_api(self, sen):
        tag = self.ner_tag_api(sen)
        B = tag_extract(sen, tag, "B")
        S = tag_extract(sen, tag, "S")

        car_info = {}
        car_info["serie"] = []
        car_info["color"] = []
        car_info["model"] = []

        # extract the car serie
        for s in S:
            label = 0
            for b in B:
                if s.lower() in self.__kg["B"]["map"][b]:
                    car_info["serie"].append(b + s)
                    label = 1
                    break
            if label == 0:
                car_info["serie"].append(s)

        for b in B:
            label = 0
            for i in car_info["serie"]:
                if b in i:
                    label = 1
                    break
            if label == 0:
                car_info["serie"].append(b)

        # extract the car model
        Y = tag_extract(sen, tag, "Y")
        N = tag_extract(sen, tag, "N")
        E = tag_extract(sen, tag, "E")
        G = tag_extract(sen, tag, "G")
        D = tag_extract(sen, tag, "D")
        Q = tag_extract(sen, tag, "Q")
        I = tag_extract(sen, tag, "I")
        M = tag_extract(sen, tag, "M")

        car_info["model"].extend(Y)
        car_info["model"].extend(N)
        car_info["model"].extend(E)
        car_info["model"].extend(G)
        car_info["model"].extend(D)
        car_info["model"].extend(Q)
        car_info["model"].extend(I)
        car_info["model"].extend(M)

        # extract the car color
        C = tag_extract(sen, tag, "C")
        car_info["color"].extend(C)
        return car_info


# main function
if __name__ == "__main__":
    kg_file = "../data/kg.json"
    Ner = Ner_tag(kg_file)
    sen = "宝马x3红色豪华版2.0t4座;奥迪a6豪华版"
    car_info = Ner.ner_log_api(sen)
    print(car_info)
