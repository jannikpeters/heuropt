import numpy as np


class TTSP:
    def __init__(self, file):
        fp = open(file, 'r')

        line = fp.readline()
        self.problem_name = line.split(':')[1].strip()
        line = fp.readline()
        self.data_type = line.split(':')[1].strip()
        line = fp.readline()
        self.dim = int(line.split(':')[1].strip())
        line = fp.readline()
        self.item_num = int(line.split(':')[1].strip())
        line = fp.readline()
        self.knapsack_capacity = int(line.split(':')[1].strip())
        line = fp.readline()
        self.min_speed = float(line.split(':')[1].strip())
        line = fp.readline()
        self.max_speed = float(line.split(':')[1].strip())
        line = fp.readline()
        self.renting_ratio = float(line.split(':')[1].strip())
        line = fp.readline()
        self.edge_weight_type = line.split(':')[1].strip()

        # read node coords
        line = fp.readline()
        self.node_coord = [(0.0, 0.0)] * self.dim
        for i in range(self.dim):
            line = fp.readline()
            temp = line.split('\t')
            self.node_coord[i] = (float(temp[1]), float(temp[2]))

        # read items
        line = fp.readline()
        self.item_weight = np.zeros(self.item_num)
        self.item_profit = np.zeros(self.item_num)
        self.item_node = np.zeros(self.item_num)

        for i in range(self.item_num):
            line = fp.readline()
            temp = line.split('\t')
            self.item_profit[i] = int(temp[1])
            self.item_weight[i] = int(temp[2])
            self.item_node[i] = int(temp[3])

    def evaluate_solution(self, tour: list, packing_list: np.ndarray):
        pass
