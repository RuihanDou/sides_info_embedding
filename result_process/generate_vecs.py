import numpy as np
from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))

from tensorflow import keras
import pandas as pd
from models.SideInfoEmbedding import SideInfoEmbedding
from utils.access_tool import *

def extract_side_info_vectors(embedding_model: SideInfoEmbedding
                           , tokenizer: keras.preprocessing.text.Tokenizer
                           , csv_path: str = conf.get_str('side_info_vec_csv_path ')
                           , tag_dict_path: str = conf.get_str('side_info_tag_vec_dict_path')
                           , id_dict_path: str = conf.get_str('side_info_id_vec_dict_path')) -> pd.DataFrame:
    """
    从训练完成的 模型中抽取出 side info 的 embedding矢量
    :param embedding_model: embedding 模型
    :param tokenizer: side info 的分词器
    :param csv_path: 抽取的向量存储csv 文件的路径
    :param tag_dict_path: side info 的 tag -> vec 的 dict 存储路径
    :param id_dict_path: side info 的 id -> vec 的 dict 存储路径
    :return:
    """
    embedding_arrays = embedding_model.get_layer(conf.get_str('layer_name')).get_weights()[0]
    id_2_tag_dict = tokenizer.index_word
    if len(id_2_tag_dict) + 1 != embedding_arrays.shape[0]:
        raise ValueError("The embedding model is not match the tokenizer !")
    tags = []
    ids = []
    vecs = []
    side_info_tag_dict = {}
    side_info_id_dict = {}
    for i in range(1, embedding_arrays.shape[0]):
        ids.append(i)
        tags.append(id_2_tag_dict[i])
        vecs.append(embedding_arrays[i])
        side_info_tag_dict[id_2_tag_dict[i]] = embedding_arrays[i]
        side_info_id_dict[i] = embedding_arrays[i]
    df = pd.DataFrame(
        {'id': ids, 'side_info_tag': tags, 'vectors': ['\t'.join([str(i) for i in vec.tolist()]) for vec in vecs]})
    df.to_csv(csv_path)
    save_dict(id_dict_path, side_info_id_dict)
    save_dict(tag_dict_path, side_info_tag_dict)
    return df

def extract_item_vectors(embedding_model: SideInfoEmbedding
                         , item_id_side_info_dict: dict = load_dict(conf.get_str('neg_samp_id_side_info_work_path'))
                         , id_to_item_dict: dict = load_dict(conf.get_str('neg_samp_id_to_item_path'))
                         , csv_path: str = conf.get_str('item_vec_csv_path ')
                         , item_dict_path: str = conf.get_str('item_vec_dict_path')
                         , id_dict_path: str = conf.get_str('item_id_vec_dict_path')) -> pd.DataFrame:
    """
    从训练完成的 模型中抽取 side info 的矢量 组合成 item 的矢量
    :param embedding_model: embedding 模型
    :param item_id_side_info_dict:  item -> side_info 的 dict
    :param id_to_item_dict:  item id -> item 的 dict
    :param csv_path: 抽取的向量存储csv 文件的路径
    :param item_dict_path: item 的 tag -> vec 的 dict 存储路径
    :param id_dict_path: item 的 id -> vec 的 dict 存储路径
    :return:
    """
    embedding_arrays = embedding_model.get_layer(embedding_model.layer_name).get_weights()[0]
    items = []
    ids = []
    vecs = []
    item_vec_dict = {}
    item_id_vec_dict = {}
    for i, tags in item_id_side_info_dict.items():
        vec = np.zeros(conf.get_int('embedding_dim'))
        for tag_id in tags:
            if tag_id <= 0:
                continue
            vec = vec + embedding_arrays[tag_id]
        items.append(id_to_item_dict[i])
        ids.append(i)
        vecs.append(vec)
        item_vec_dict[id_to_item_dict[i]] = vec
        item_id_vec_dict[i] = vec
    df = pd.DataFrame(
        {'id': ids, 'item': items, 'vectors': ['\t'.join([str(i) for i in vec.tolist()]) for vec in vecs]})
    df.to_csv(csv_path)
    save_dict(item_dict_path, item_vec_dict)
    save_dict(id_dict_path, item_id_vec_dict)
    return df


