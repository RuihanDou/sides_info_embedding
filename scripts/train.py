from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))

"""
从 configure 里解析 训练的配置
"""

side_infomation_path = conf.get_string('side_infomation_path')
# 序列表格路径
sequences_path = conf.get_string('side_infomation_path')
# 读取 keras.Tokenizer 的路径, 该tokenizer 把 side_information 换成 embedding 矢量的查询索引
tokenizer_path = conf.get_string('tokenizer_path')
# item 对应side info
side_info_dict_path = conf.get_string('side_info_dict_path')
# 连续整数空间中id 对应side info
neg_samp_id_side_info_work_path = conf.get_string('neg_samp_id_side_info_work_path')
# 存储图的路径
graph_file_path = conf.get_string('graph_file_path')
# 存储一个epoch随机游走序列路径
walk_sequence_file_path = conf.get_string('walk_sequence_file_path')
# item 转化为 连续整数空间中id 的映射
item_to_neg_samp_id_path = conf.get_string('item_to_neg_samp_id_path')
# 连续整数空间中id 转化为 item 的映射
neg_samp_id_to_item_path = conf.get_string('neg_samp_id_to_item_path')
# 设置本地pysark session配置
excecutor_memory = conf.get_string('excecutor_memory')
driver_memory = conf.get_string('driver_memory')
max_result_size = conf.get_string('max_result_size')
# item 的 tag (side info) 的种类 个数 + [UNK]
side_info_tag_size = conf.get_int('side_info_tag_size')
# 单条item 的 tag (side info) 最多个数 为了方便 219 --> 220
side_info_max_num_tags = conf.get_int('side_info_max_num_tags')
# 每条tag的 embedding size 大小
embedding_dim = conf.get_int('embedding_dim')
# tag 的 category
side_info_category = conf.get_list('side_info_category')
# sequences 的 category
sequence_category = conf.get_string('sequence_category')
# 随机游走生成的最长序列长度
max_walk_seq_length = conf.get_int('max_walk_seq_length')
# 随机游走生成的最短序列长度
min_walk_seq_length = conf.get_int('min_walk_seq_length')
# 终止游走概率
walk_end_probability = conf.get_float('walk_end_probability')
# 图的节点数(方便调试用)
vertices_num = conf.get_int('vertices_num')
window_size = conf.get_int('window_size')
negative_sample_rate = conf.get_float('negative_sample_rate')
# 训练批大小
batch_size = conf.get_int('batch_size')
# shuffle 用的缓存大小
buffer_size = conf.get_int('buffer_size')
# embedding layer 的名称
layer_name = conf.get_string('layer_name')
# 存储训练完成的 side_info 的 csv 文件 包含三列 id， tag， vec
side_info_vec_csv_path = conf.get_string('side_info_vec_csv_path')
# 存储训练完成后的 side_info 的 tag -> vec 的dict
side_info_tag_vec_dict_path = conf.get_string('side_info_tag_vec_dict_path')
# 存储训练完成后的 side_info 的 id -> vec 的dict
side_info_id_vec_dict_path = conf.get_string('side_info_id_vec_dict_path')
# 存储训练完成的 item 的 csv 文件 包含三列 id， item， vec
item_vec_csv_path = conf.get_string('item_vec_csv_path')
# 存储训练完成后的 item 的 item -> vec 的dict
item_vec_dict_path = conf.get_string('item_vec_dict_path')
# 存储训练完成后的 item 的 id -> vec 的dict
item_id_vec_dict_path = conf.get_string('item_id_vec_dict_path')