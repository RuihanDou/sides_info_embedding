from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))

from pyspark.sql import *
import pickle as pkl
from utils.tokenizer_tool import load_tokenizer
from tensorflow import keras
from utils.access_tool import *
import tensorflow as tf


def get_side_information_dict(side_info_path: str = conf.get_string('side_infomation_path')
                              , tokenizer: keras.preprocessing.text.Tokenizer = load_tokenizer( conf.get_string('tokenizer_path'))
                              , padding_max_size: int = conf.get_int('side_info_max_num_tags')
                              , catagory_list: list = conf.get_list('side_info_category')) -> dict:
    """
        获取 embedding 使用的 side info

    :param side_info_path: csv 格式的 side info 表格存储路径
    :param tokenizer: side info 使用的分词器，把 side info 的每一条 tag 映射为对应的连续区间的 id， id的连续区间为[1, side_info_tag_size-1]
    :param padding_max_size: 调控参数，由于本项目对多余tag使用mask，所以 padding_max_size 大于或等于 item 中最多tag数
    :param catagory_list: 从 csv 文件中抽取哪些列作为用于embedding的 side info对象
    :return: side info 的 dict {key -> item : value -> [side_info_list]}
    """
    spark = SparkSession.builder.master('local').appName("side_information") \
        .config('spark.excecutor.memory', conf.get_string('excecutor_memory')) \
        .config('spark.driver.memory', conf.get_string('driver_memory')) \
        .config('spark.driver.maxResultSize', conf.get_string('max_result_size')) \
        .getOrCreate()
    df = spark.read.options(header='True', inferSchema='True').csv(side_info_path)
    side_info_dict = {}
    for row in df.collect():
        sales_id = row['sales_id']
        item_text = ""
        for cata in catagory_list:
            item_text = item_text + "," + str(row[cata])
        tags = tokenizer.texts_to_sequences([item_text])[0]
        tags = tags + (padding_max_size - len(tags)) * [0]
        side_info_dict[sales_id] = tags
    spark.stop()
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

def get_side_info_mask(neg_samp_id_side_info_dict: dict = load_dict(conf.get_string('neg_samp_id_to_side_info_dict_path'))):
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

    # pickle 读取 dict
    side_info_dict = {}
    with open(conf.get_string('side_info_dict_path'), 'rb') as handle:
        side_info_dict = pkl.load(handle)

    # pickle 读取 item_to_neg_samp_id dict
    item_2_id_dict = {}
    with open(conf.get_string('item_to_neg_samp_id_path'), 'rb') as handle:
        item_2_id_dict = pkl.load(handle)

    neg_samp_id_side_info_dict = get_neg_samp_id_side_info_dict(item_2_id_dict, side_info_dict)
    with open(conf.get_string('neg_samp_id_to_side_info_dict_path'), 'wb') as handle:
        pkl.dump(neg_samp_id_side_info_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print("success.")

    print(neg_samp_id_side_info_dict)
    # catagory_list = conf.get_list('side_info_category')
    # print(catagory_list)

    # side_info_dict = get_side_information_dict()
    # # pickle 存储 dict
    # with open(os.path.join('data', 'side_info_dict.pickle'), 'wb') as handle:
    #     pkl.dump(side_info_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # # pickle 读取 dict
    # read_dict = {}
    # with open(os.path.join('data', 'side_info_dict.pickle'), 'rb') as handle:
    #     read_dict = pkl.load(handle)
    #
    # print("OK")
    # side_info_path = os.path.join(conf.get_string('work_path'), conf.get_string('side_infomation_path'))
    # tokenizer = load_tokenzier(os.path.join(conf.get_string('work_path'), conf.get_string('tokenizer_path')))
    # spark = SparkSession.builder.master('local').appName("side_information") \
    #     .config('spark.excecutor.memory', conf.get_string('excecutor_memory')) \
    #     .config('spark.driver.memory', conf.get_string('driver_memory')) \
    #     .config('spark.driver.maxResultSize', conf.get_string('max_result_size')) \
    #     .getOrCreate()
    # df = spark.read.options(header='True', inferSchema='True').csv(side_info_path)
    # side_info_dict = {}
    # max_padding_size = 0
    # for row in df.collect():
    #     sales_id = row['sales_id']
    #     item_text = ""
    #     for cata in catagory_list:
    #         item_text = item_text + "," + str(row[cata])
    #     if row['visa_city'] is not None:
    #         print('visa_city : ' + row['visa_city'])
    #
    #
    #     tags = tokenizer.texts_to_sequences([item_text])[0]
    #     max_padding_size = max(len(tags), max_padding_size)
    #
    # print(max_padding_size)
    # spark.stop()

