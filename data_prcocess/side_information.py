from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))

from pyspark.sql import *
import pickle as pkl
from utils.tokenizer_tool import load_tokenzier
from tensorflow import keras
from utils.access_tool import *
import tensorflow as tf


def get_side_information_dict(side_info_path: str = os.path.join(conf.get_string('work_path'), conf.get_string('side_infomation_path')),
                              tokenizer: keras.preprocessing.text.Tokenizer = load_tokenzier(os.path.join(conf.get_string('work_path'), conf.get_string('tokenizer_path'))),
                              padding_max_size: int = conf.get_int('side_info_max_num_tags'), catagory_list: list = conf.get_list('side_info_category')) -> dict:
    """
    该函数获取 side_info 的 dict

    :param side_info_path: 存储 side information 文件的路径，默认文件是 csv 文件
    :param tokenizer: 把 side information 映射对应为 index 的分词工具
    :param padding_max_size: 每条item最多使用 padding_max_size条 side-info-tags 表示
    :param catagory_list:  从 side information 选择side_information 的类目
    :return:
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

def get_neg_samp_id_side_info(item_to_neg_samp_id_dict: dict, side_info_dict: dict) ->dict:
    neg_samp_id_side_info_dict = {}
    for item, id in item_to_neg_samp_id_dict.items():
        # 有可能有没有统计到side info 的 item
        if item not in side_info_dict:
            neg_samp_id_side_info_dict[id] = [0] * conf.get_int('side_info_max_num_tags')
        else:
            neg_samp_id_side_info_dict[id] = side_info_dict[item]
    return neg_samp_id_side_info_dict

def get_set_info_tensor(neg_samp_id_side_info_dict: dict = load_dict(conf.get_string('neg_samp_id_side_info_work_path'))):
    const_list = []
    for k, v in neg_samp_id_side_info_dict.items():
        const_list.append(v)
    side_info_constant_tensor = tf.constant(const_list, dtype=tf.int64)
    return side_info_constant_tensor

def get_set_info_mask(neg_samp_id_side_info_dict: dict = load_dict(conf.get_string('neg_samp_id_side_info_work_path'))):
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

    neg_samp_id_side_info_dict = get_neg_samp_id_side_info(item_2_id_dict, side_info_dict)
    with open(conf.get_string('neg_samp_id_side_info_work_path'), 'wb') as handle:
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

