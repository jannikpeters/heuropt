from functools import lru_cache

import numpy as np


class TTSP:

    @lru_cache()
    def dist(self, first, second):
        return np.ceil(np.sqrt((self.node_coord[first,0]-self.node_coord[second,0])**2 + (self.node_coord[first,1]-self.node_coord[second,1])**2))


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
        self.node_coord = np.zeros((self.dim,2))
        for i in range(self.dim):
            line = fp.readline()
            temp = line.split('\t')
            self.node_coord[i,0] = int(temp[1])
            self.node_coord[i,1] = int(temp[2])

        # read items
        line = fp.readline()
        self.item_weight = np.zeros(self.item_num, dtype=int)
        self.item_profit = np.zeros(self.item_num, dtype=int)
        self.item_node = np.zeros(self.item_num, dtype=int)

        for i in range(self.item_num):
            line = fp.readline()
            temp = line.split('\t')
            self.item_profit[i] = int(temp[1])
            self.item_weight[i] = int(temp[2])
            self.item_node[i] = int(temp[3])-1

        # Additional for faster computation
        self.indexes_items_in_city = np.array([np.where(self.item_node == city_i)[0] for city_i in \
                range(self.dim)])



