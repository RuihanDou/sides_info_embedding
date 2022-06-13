from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'config.conf'))
os.chdir(conf.get_string('work_path'))
import sys
sys.path.append(conf.get_string('work_path'))
sequence_category = conf.get_string('sequence_category')
import pandas as pd
from tqdm import tqdm


def sequences_dataframe_to_graph_pair_file(sequences_path: str
                                           , graph_file_path: str):
    """
    把点击序列文件，转化为带权图文件
    :param sequences_path:  点击序列 csv 文件地址路径
    :param graph_file_path:   存储的带权图 txt 文件路径
    :return:
    """
    print("load basic sequence file from csv file.")
    df = pd.read_csv(sequences_path)
    # sequence_series 是 pandas.core.series.Series 类型的数据，不能支持用 .iterrows() 遍历可以
    # sequence_series 支持直接遍历
    sequence_series = df[sequence_category]
    graph_dict = {}
    items_set = set()
    for seq in tqdm(sequence_series):
        items = [int(item) for item in str(seq).split('\t')]
        # 序列长度为 1 无法构建图
        if len(items) <= 1:
            continue
        # 序列长度 >= 2 时
        seq_len = len(items)
        for i in range(1, seq_len):
            # 排除非法节点
            if items[i-1] <= 0 or items[i] <= 0:
                continue
            # 排除 自 环
            if items[i-1] == items[i]:
                continue
            items_set.add(items[i-1])
            items_set.add(items[i])
            pair = (min(items[i-1], items[i]), max(items[i-1], items[i]))
            if pair in graph_dict:
                graph_dict[pair] += 1
            else:
                graph_dict[pair] = 1
    vertex_edge_formation = "{} {}\n"
    graph_file_line_formation = "{} {} {}\n"
    with open(graph_file_path, 'w') as file:
        file.write(vertex_edge_formation.format(items_set.__len__(), graph_dict.__len__()))
        for pair in graph_dict:
            file.write(graph_file_line_formation.format(pair[0], pair[1], graph_dict[pair]))







if __name__ == '__main__':
    sequences_dataframe_to_graph_pair_file(sequences_path=conf.get_string('sequences_path'),
                                           graph_file_path=conf.get_string('graph_file_path'))