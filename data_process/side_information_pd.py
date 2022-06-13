from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'config.conf'))
os.chdir(conf.get_string('work_path'))
import sys
sys.path.append(conf.get_string('work_path'))

import pandas as pd
from tqdm import tqdm
import pickle as pkl
from utils.tokenizer_tool import load_tokenizer
from tensorflow import keras
from utils.access_tool import *
import tensorflow as tf


def get_side_information_dict(side_info_path: str
                              , tokenizer: keras.preprocessing.text.Tokenizer
                              , padding_max_size: int
                              , catagory_list: list) -> dict:
    """
        获取 embedding 使用的 side info

    :param side_info_path: csv 格式的 side info 表格存储路径
    :param tokenizer: side info 使用的分词器，把 side info 的每一条 tag 映射为对应的连续区间的 id， id的连续区间为[1, side_info_tag_size-1]
    :param padding_max_size: 调控参数，由于本项目对多余tag使用mask，所以 padding_max_size 大于或等于 item 中最多tag数
    :param catagory_list: 从 csv 文件中抽取哪些列作为用于embedding的 side info对象
    :return: side info 的 dict {key -> item : value -> [side_info_list]}
    """
    print("load basic side infomation data from csv file.")
    df = pd.read_csv(side_info_path, low_memory=False)
    side_info_dict = {}
    for i, row in tqdm(df.iterrows()):
        sales_id = row['sales_id']
        item_text = ""
        for cata in catagory_list:
            item_text = item_text + "," + str(row[cata])
        tags = tokenizer.texts_to_sequences([item_text])[0]
        tags = tags + (padding_max_size - len(tags)) * [0]
        side_info_dict[sales_id] = tags
    return side_info_dict

def get_neg_samp_id_side_info_dict(item_to_neg_samp_id_dict: dict, side_info_dict: dict) ->dict:
    """
        把 side info 的 dict {key -> item : value -> [side_info_list]}
    转化为 {key -> item id : value -> [side_info_list]}

        其中 item id 为一个连续区间的整型数集合 [0, vertices_num - 1] 注意，与 side_info 使用 [UNK] 作为 id = 0的占位不同，
    item id 的 0 是有实体 item 对应的

    :param item_to_neg_samp_id_dict: item 到 item id 映射的 dict {key -> item : value -> item id}
    :param side_info_dict: side info 的 dict {key -> item : value -> [side_info_list]}
    :return: {key -> item id : value -> [side_info_list]}
    """
    neg_samp_id_side_info_dict = {}
    for item, id in item_to_neg_samp_id_dict.items():
        # 有可能有没有统计到side info 的 item
        if item not in side_info_dict:
            neg_samp_id_side_info_dict[id] = [0] * conf.get_int('side_info_max_num_tags')
        else:
            neg_samp_id_side_info_dict[id] = side_info_dict[item]
    return neg_samp_id_side_info_dict

def get_side_info_tensor(neg_samp_id_side_info_dict: dict):
    """
        生成 item id 为索引的，索引下为 该 item 对应 side info 对应 id 的矢量 的 tensor

        tensor.shape = (item num, side_info_max_num_tags)

    :param neg_samp_id_side_info_dict: {key -> item id : value -> [side_info_list]}
    :return: tf.Tensor with shape (item num, side_info_max_num_tags)
    """
    const_list = []
    for k, v in neg_samp_id_side_info_dict.items():
        const_list.append(v)
    side_info_constant_tensor = tf.constant(const_list, dtype=tf.int64)
    return side_info_constant_tensor

def get_side_info_mask(neg_samp_id_side_info_dict: dict):
    """
        生成 item id 为索引的，索引下为 该 item 对应 side info 对应 id 的矢量 的 mask

        tensor.shape = (item num, side_info_max_num_tags)

        当 side info vector 在 第 i 位上是 0 ， 那么 side info mask 在该位上是 0 ， 否则 为 1

    :param neg_samp_id_side_info_dict: neg_samp_id_side_info_dict: {key -> item id : value -> [side_info_list]}
    :return: tf.Tensor with shape (item num, side_info_max_num_tags)
    """
    const_list = []
    for k, v in neg_samp_id_side_info_dict.items():
        const_list.append(v)
    side_info_constant_tensor = tf.constant(const_list, dtype=tf.int64)
    mask = tf.cast(tf.math.not_equal(side_info_constant_tensor, 0), tf.float32)
    return mask


if __name__ == '__main__':
    side_info_dict = get_side_information_dict(side_info_path=conf.get_string('side_infomation_path')
                                               , tokenizer=load_tokenizer(conf.get_string('tokenizer_path'))
                                               , padding_max_size=conf.get_int('side_info_max_num_tags')
                                               , catagory_list=conf.get_list('side_info_category'))
    print(side_info_dict)