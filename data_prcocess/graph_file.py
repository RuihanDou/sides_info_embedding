from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
from pyspark.sql import *
sequence_category = conf.get_string('sequence_category')

def sequences_dataframe_to_graph_pair_file(sequences_path: str = conf.get_string('sequences_path'),graph_file_path: str = conf.get_string('graph_file_path')):
    spark = SparkSession.builder.master('local').appName("side_information") \
        .config('spark.excecutor.memory', conf.get_string('excecutor_memory')) \
        .config('spark.driver.memory', conf.get_string('driver_memory')) \
        .config('spark.driver.maxResultSize', conf.get_string('max_result_size')) \
        .getOrCreate()
    df = spark.read.options(header='True', inferSchema='True').csv(sequences_path).select(sequence_category)
    graph_dict = {}
    items_set = set()
    for row in df.collect():
        items = [int(item) for item in str(row[sequence_category]).split('\t')]
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
    spark.stop()
    vertex_edge_formation = "{} {}\n"
    graph_file_line_formation = "{} {} {}\n"
    with open(graph_file_path, 'w') as file:
        file.write(vertex_edge_formation.format(items_set.__len__(), graph_dict.__len__()))
        for pair in graph_dict:
            file.write(graph_file_line_formation.format(pair[0], pair[1], graph_dict[pair]))




if __name__ == '__main__':

    sequences_dataframe_to_graph_pair_file()

    # spark = SparkSession.builder.master('local').appName("side_information") \
    #     .config('spark.excecutor.memory', conf.get_string('excecutor_memory')) \
    #     .config('spark.driver.memory', conf.get_string('driver_memory')) \
    #     .config('spark.driver.maxResultSize', conf.get_string('max_result_size')) \
    #     .getOrCreate()
    # df = spark.read.options(header='True', inferSchema='True').csv(conf.get_string('sequences_path')).select(sequence_category)
    # for row in df.collect():
    #     items = [int(item) for item in str(row[sequence_category]).split('\t')]
    #     if len(items) :
    #         print(items)
    #         break
    # spark.stop()