import random
from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
from data_process.asymmetrical_weighted_graph import AsymmetricalWeightedGraph
import pickle as pkl


class RandomWalkSequenceGenerator:
    """
    使用 权重非对称图进行 随机游走，生成采样序列
    """
    def __init__(self, asymmerical_weighted_graph: AsymmetricalWeightedGraph
                 , max_walk_seq_length: int = conf.get_int('max_walk_seq_length')
                 , min_walk_seq_length: int = conf.get_int('min_walk_seq_length')
                 , walk_sequence_file_path: str = conf.get_string('walk_sequence_file_path')
                 , walk_end_probability: float = conf.get_float('walk_end_probability')
                 , item_to_neg_samp_id_path: str = conf.get_string('item_to_neg_samp_id_path')
                 , neg_samp_id_to_item_path: str = conf.get_string('neg_samp_id_to_item_path')):
        """
            该 class 的主要任务是对图进行随机游走生成序列，次要任务为生成 item 映射到 连续整数空间的 id 其中id的范围为

                            [0, vertices_num - 1]

            该 class 生成一条游走序列的 方法是 generate_a_sequence(start_id) 其中 start_id 为 item id
            该 class 生成一次训练 epoch 的游走序列并存储 为 txt 文件的方法是 generate_epoch() 产生的序列覆盖图中所有 vertex (item)

            * item id (或 vertex id) 是从 0 开始计的， 一共有 graph.V() 个，即 vertices_num 个

            * 序列内容量约定为 (该序列去重v的个数 / 该序列长度) * 1.0

            * 由于 每 epoch 需要重新生成游走序列，该 class 提供清理掉本次游走的方法 clean_epoch()

        :param asymmerical_weighted_graph: 经过 asymmetry() 后的 AsymmetricalWeightedGraph
        :param max_walk_seq_length: 游走生成序列的最大长度
        :param min_walk_seq_length: 游走序列生成的约定最小长度，当该序列的内容量小于0.1时，允许裁剪序列小于该设定
        :param walk_sequence_file_path: 生成序列存储txt文件的路径
        :param walk_end_probability: 调控序列增长的参数，在当前v终止掉序列增长的概率
        :param item_to_neg_samp_id_path: dict {key -> item : value -> item id} 的存储地址
        :param neg_samp_id_to_item_path: dict {key -> item id : value -> item} 的存储地址
        """
        self.graph = asymmerical_weighted_graph
        self.items_list = [int(item) for item in self.graph.get_walk_map().keys()]
        self.max_walk_seq_length = max_walk_seq_length
        self.min_walk_seq_length = min_walk_seq_length
        self.walk_end_probability = walk_end_probability
        self.items_num = self.graph.V()
        self.id2items_map = {}
        self.items2id_map = {}
        self.walk_sequence_file_path = walk_sequence_file_path
        self.item_to_neg_samp_id_path = item_to_neg_samp_id_path
        self.neg_samp_id_to_item_path = neg_samp_id_to_item_path
        for i, item in enumerate(self.items_list):
            if i >= self.items_num:
                raise ValueError("Wrong item id")
            self.id2items_map[i] = item
            self.items2id_map[item] = i
        self.visited = set()
        self.save_item_encoder_decoder()

    def save_item_encoder_decoder(self):
        with open(self.item_to_neg_samp_id_path, 'wb') as handle:
            pkl.dump(self.items2id_map, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(self.neg_samp_id_to_item_path, 'wb') as handle:
            pkl.dump(self.id2items_map, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def generate_a_sequence(self, start_id):
        curr = self.id2items_map[start_id]
        seq = [curr]
        # seq 长度不大于 max_walk_seq_length
        while(len(seq) < self.max_walk_seq_length):
            # 当 seq 长度超过最小限制 min_walk_seq_length 有 1 / 20 的概率会终止序列
            if(len(seq) >= self.min_walk_seq_length):
                dice = random.random()
                if dice > (1.0 - self.walk_end_probability):
                    break
            nextt = self._add_next(seq, curr)
            curr = nextt
        return seq

    def _add_next(self, seq, curr):
        """
            游走到下一步的逻辑: 掷出一个随机数 dice
        遍历 cur 的邻接 map，map 的 value 升序， 当 value >= dice 则下一步走向对应的 key

            * 下列逻辑中 key = item, value = weight

        :param seq:
        :param curr:
        :return:
        """
        adjacent_map = self.graph.get_walk_map()[curr]
        dice = random.random()
        nextt = None
        for item, weight in adjacent_map.items():
            if weight >= dice:
                nextt = item
                seq.append(nextt)
                return nextt
        raise ValueError("Wrong Walk Map.")

    def generate_epoch(self):
        visited = set()
        with open(self.walk_sequence_file_path, 'w') as f:
            print("Begin Random Walk On The Graph")
            seq = self.generate_a_sequence(random.randint(0, self.items_num))
            f.write('\t'.join([str(it) for it in seq]) + '\n')
            # count 用于统计展示完成度
            count = 0
            while(True):
                for item in seq:
                   visited.add(self.items2id_map[item])
                if count % 10000 == 0:
                    print(str(round(visited.__len__() / self.items_num * 100, 2)) + "% ready")
                seq_start = -1
                for id in range(self.items_num):
                    if id not in visited:
                        seq_start = id
                        break
                if seq_start == -1:
                    break
                seq = self.generate_a_sequence(seq_start)
                # 处理稀疏序列，当序列长度 远远大于 序列内容（序列中包含的非重复元素个数）时
                unique_item_set = set(seq)
                rate = (len(unique_item_set) / len(seq)) * 1.0
                if rate < 0.1:
                    seq = list(unique_item_set) * 4
                count += 1
                f.write('\t'.join([str(it) for it in seq]) + '\n')
            print("Random Walk Generate An Epoch Sequence.")

    def clean_epoch(self):
        os.remove(self.walk_sequence_file_path)


if __name__ == '__main__':

    graph_file_path = conf.get_string('graph_file_path')
    graph = AsymmetricalWeightedGraph(graph_file_path)
    seq_generator = RandomWalkSequenceGenerator(asymmerical_weighted_graph=graph)
    seqs = seq_generator.generate_epoch()
    print("successful")

