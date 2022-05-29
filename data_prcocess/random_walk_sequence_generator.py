import random
from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
from  asymmetrical_weighted_graph import AsymmetricalWeightedGraph



class RandomWalkSequenceGenerator:
    """"""
    def __init__(self, asymmerical_weighted_graph: AsymmetricalWeightedGraph
                 , max_walk_seq_length: int = conf.get_int('max_walk_seq_length')
                 , min_walk_seq_length: int = conf.get_int('min_walk_seq_length')
                 , walk_sequence_file_path: str = conf.get_string('walk_sequence_file_path')):
        self.graph = asymmerical_weighted_graph
        self.items_list = [int(item) for item in self.graph.walk_map.keys()]
        self.max_walk_seq_length = max_walk_seq_length
        self.min_walk_seq_length = min_walk_seq_length
        self.items_num = self.graph.V()
        self.id2items_map = {}
        self.items2id_map = {}
        self.walk_sequence_file_path = walk_sequence_file_path
        for i, item in enumerate(self.items_list):
            if i >= self.items_num:
                raise ValueError("Wrong item id")
            self.id2items_map[i] = item
            self.items2id_map[item] = i
        self.visited = set()

    def generate_a_sequence(self, start_id):
        curr = self.id2items_map[start_id]
        seq = [curr]
        # seq 长度不大于 max_walk_seq_length
        while(len(seq) < self.max_walk_seq_length):
            # 当 seq 长度超过最小限制 min_walk_seq_length 有 1 / 20 的概率会终止序列
            if(len(seq) >= self.min_walk_seq_length):
                dice = random.random()
                if dice > 0.95:
                    break
            nextt = self._add_next(seq, curr)
            curr = nextt
        return seq

    def _add_next(self, seq, curr):
        adjacent_map = self.graph.walk_map[curr]
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
            seq = self.generate_a_sequence(random.randint(0, self.items_num))
            f.write('\t'.join([str(it) for it in seq]) + '\n')
            while(True):
                for item in seq:
                   visited.add(self.items2id_map[item])
                seq_start = -1
                for id in range(self.items_num):
                    if id not in visited:
                        seq_start = id
                        break
                if seq_start == -1:
                    break
                seq = self.generate_a_sequence(seq_start)
                f.write('\t'.join([str(it) for it in seq]) + '\n')






if __name__ == '__main__':
    graph_file_path = conf.get_string('graph_file_path')
    graph = AsymmetricalWeightedGraph(graph_file_path)
    seq_generator = RandomWalkSequenceGenerator(asymmerical_weighted_graph=graph)

    seqs = seq_generator.generate_epoch()
    print("successful")

