import random

from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
import sys
sys.path.append(conf.get_string('work_path'))
from utils.access_tool import *
from tqdm import tqdm
import numpy as np

import tensorflow as tf


def generate_pairs_by_skip_gram(sequence: list
                                , vertex_num: int = conf.get_int('vertices_num')
                                , window_size: int = conf.get_int('window_size')
                                , negative_sample_rate:float = conf.get_float('negative_sample_rate')):
    """
        设 图的节点数量为 V

        本函数注意：

        *由于本项目设置的 item_id 的区间为 [0, V-1]
            采样函数 是 [1, V], 所以先给 id + 1 处理，经过采样之后，再给所有 id - 1

    :param sequence: list(id) 注意是 item id(连续空间的) 不是 item
    :param vertex_num: 图 vertex (item) 的数量
    :param window_size: 窗大小
    :param negative_sample_rate: 负样本数 / 正样本数 比例

    :return: target_ids, context_ids, labels 三个列表的元组
    """
    # id + 1 操作
    sequence_ = [i + 1 for i in sequence]
    pairs, labels = tf.keras.preprocessing.sequence.skipgrams(sequence_, vocabulary_size=vertex_num + 1,
                                                              window_size=window_size,
                                                              negative_samples=negative_sample_rate, shuffle=True)
    targets = []
    contexts = []
    target_context_labels = []
    for (target, context), label in zip(pairs, labels):
        # id - 1 操作
        targets.append(target - 1)
        contexts.append(context - 1)
        target_context_labels.append(label)

    return targets, contexts, target_context_labels

def generate_train_epoch_dataset(walk_sequence_path: str = conf.get_string('walk_sequence_file_path'),
                                item_to_id_dict: dict = {}, vertex_num: int = conf.get_int('vertices_num'),
                                window_size: int = conf.get_int('window_size'),
                                negative_sample_rate: float = conf.get_float('negative_sample_rate'),
                                batch_size=conf.get_int('batch_size'),
                                buffer_size=conf.get_int('buffer_size')) -> tf.data.Dataset:
    """

        本函数读入 一个epoch 的 随机游走序列 txt 文件 ，并转化为 ((target, context), label) 的 tf.Dataset文件

    :param walk_sequence_path:
    :param item_to_id_dict:
    :param vertex_num:
    :param window_size:
    :param negative_sample_rate:
    :param batch_size:
    :param buffer_size:
    :return:
    """
    lines = None
    with open(walk_sequence_path, 'r') as file:
        lines = file.readlines()
    lines = [[item_to_id_dict[int(i)] for i in line.replace('\n', '').split('\t')] for line in lines]
    # format_lines = []
    # for line in lines:
    #     line_list = line.replace('\n', '').split('\t')
    #     format_line = []
    #     if len(line_list) > 1:
    #         for token in line_list:
    #             if int(token) in item_to_id_dict:
    #                 format_line.append(item_to_id_dict[int(token)])
    #     if len(format_line) > 1:
    #         format_lines.append(format_line)
    # lines = format_lines

    print("Generate targets, contexts and labels for an epoch.")
    targets_data = []
    contexts_data = []
    labels_data = []
    for line in tqdm(lines):
        targets, contexts, labels = generate_pairs_by_skip_gram(line, vertex_num, window_size, negative_sample_rate)
        targets_data.extend(targets)
        contexts_data.extend(contexts)
        labels_data.extend(labels)
    print("Package targets, context and labels to dataset.")
    targets_data = np.array(targets_data)
    contexts_data = np.array(contexts_data)
    labels_data = np.array(labels_data)
    targets_data = targets_data[:, np.newaxis]
    contexts_data = contexts_data[:, np.newaxis]
    labels_data = labels_data[:, np.newaxis]
    dataset = tf.data.Dataset.from_tensor_slices(((targets_data, contexts_data), labels_data))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print("dataset ready")
    return dataset



