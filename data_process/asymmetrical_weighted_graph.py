import math

from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
import tqdm
import collections

class AsymmetricalWeightedGraph:
    """
    采样使用的权重非对称图

    双向权重不同

    无向图: A -> B 必然有 B -> A
    权重非对称: A -> B 权重 为 w1， B -> A 权重 为 w2


    """
    def __init__(self, file_path: str):
        """
        :param file_path: 读取 graph file 的地址
            graph file 的第一行 存储的是 vertex 数量 和 edge 数量
            因为 vertex 表示不是从 0 开始的连续整数数列， graph file 存储的格式是 line(v1, v2, w)
            v1 v2 是整形，w 是 序列中出现连续的 v1 v2 的次数
            其中 每一 line 表示的 edge 关系不会在 graph file 中重复
            其中 格式保证 v1 < v2
        """
        self.file_path = file_path
        print("load graph from {}".format(self.file_path))
        lines = None
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            raise ValueError('Expected something from input file!')

        # lines[0] -> V E
        self._V, self._E = (int(i) for i in lines[0].split())

        if self._V < 0:
            raise ValueError('V must be non-negative')

        if self._E < 0:
            raise ValueError('E must be non-negative')

        # self._adj 形如 {v0 : {v1 : w1, v2 : w2}, ...}
        self._adj = {}
        for each_line in tqdm.tqdm(lines[1:]):
            a, b, weight = (int(i) for i in each_line.split())
            if a == b:
                continue
            self.add_adj(a, b, weight)

        self._walk_map = {}
        self.asymmetry()

    def add_adj(self, a, b, w):
        """
        把edge (a b) 权重w添加进 图，此时没有权重非对称化，权重对称
        """
        if a not in self._adj:
            self._adj[a] = collections.OrderedDict()
        if b not in self._adj:
            self._adj[b] = collections.OrderedDict()
        self._adj[a][b] = w
        self._adj[b][a] = w

    def V(self):
        return self._V

    def E(self):
        return self._E

    def has_edge(self, v, w):
        self.validate_vertex(v)
        self.validate_vertex(w)
        return w in self._adj[v] and v in self._adj[w]

    def validate_vertex(self, v):
        if v not in self._adj:
            raise ValueError('vertex ' + str(v) + ' is invalid')

    def adj_list(self, v):
        self.validate_vertex(v)
        return list(self._adj(v).keys())

    def get_weight(self, v, w):
        """
        因为权重的非对称性，本方法为 v -> w 的权重，非 w -> v 的权重
        """
        self.validate_vertex(v)
        if w in self._adj(v):
            return self._adj[v][w]
        raise ValueError('No edge {}->{}'.format(v, w))

    def degree(self, v):
        return len(self.adj_list(v))

    def remove_edge(self, v, w):
        self.validate_vertex(v)
        self.validate_vertex(w)
        if w in self._adj[v]:
            self._adj[v].pop(w)
        if v in self._adj[w]:
            self._adj[w].pop(v)

    def __str__(self):
        res = ['V = {}, E = {}'.format(self._V, self._E)]
        for v in self._adj:
            res.append(
                '{}: {}'.format(
                    v,
                    ' '.join('{}({})'.format(w, self._adj[v][w]) for w in self._adj[v]),
                ),
            )
        return '\n'.join(res)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return AsymmetricalWeightedGraph(self._filename)

    def sort_weighted(self):
        """
        把 self._adj中每个 adj 按照权重从大到小排列
        """
        print("Sort graph weight")
        for key in tqdm.tqdm(self._adj.keys()):
            ordered_dict = self._adj[key]
            weights = list(ordered_dict.values())
            weight_set = set(weights)
            weight_set = list(weight_set)
            weight_set.sort(reverse=True)
            sorted_ordered_dict = collections.OrderedDict()
            for weight in weight_set:
                for i, w in ordered_dict.items():
                    if w == weight:
                        sorted_ordered_dict[i] = w
            self._adj[key] = sorted_ordered_dict

    def asymmetry(self):
        self.sort_weighted()
        print("Asymmetry the weight.")
        for vertex in tqdm.tqdm(self._adj.keys()):
            adjacent_vertices = self._adj[vertex]
            log_adjacent_vertices = collections.OrderedDict()
            # 对所有原有的连击 weight 做对数化处理
            for v in adjacent_vertices:
                # new_weight = ln(1 + weight)
                log_adjacent_vertices[v] = math.log1p(adjacent_vertices[v])
            # 取总量权重，方便决定随机方向
            sum_log_weights = sum(log_adjacent_vertices.values())
            accumulate = 0.0
            vertex_direct = collections.OrderedDict()
            for v, log_w in log_adjacent_vertices.items():
                director_weight = (accumulate + log_w) / sum_log_weights
                vertex_direct[v] = director_weight
                accumulate += log_w
            self._walk_map[vertex] = vertex_direct

    def get_work_map(self):
        return self._walk_map

if __name__ == '__main__':
    graph_file_path = conf.get_string('graph_file_path')
    graph = AsymmetricalWeightedGraph(graph_file_path)
    adj = graph.get_adj()
    max_degree = 0
    for map in adj.values():

        max_degree = max(map.__len__(), max_degree)

    print(max_degree)




