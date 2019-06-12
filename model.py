import itertools
import os
from functools import lru_cache
import numpy as np
from numba import jitclass
import numba as nb
from scipy.spatial.distance import pdist, squareform

class TTSP:

    def dist(self, first, second):
        return self.dist_cache[first, second]

    @lru_cache()
    def old_dist(self, first, second):
        """Deprecated: Only here for historic reason. Slow!!!"""
        man = np.ceil(np.sqrt((self.node_coord[first, 0] - self.node_coord[second, 0]) ** 2 + (
                self.node_coord[first, 1] - self.node_coord[second, 1]) ** 2))
        return man



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
        self.old_rr = self.renting_ratio

        # read node coords
        line = fp.readline()
        self.node_coord = np.zeros((self.dim,2), dtype=int)
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
        index_cache_name = 'pickles/'+file.split('/')[1]+'_index'
        if os.path.isfile(index_cache_name+'.npy'):
            print('Loading', index_cache_name, 'from Cache')
            self.indexes_items_in_city = np.load(index_cache_name+'.npy', allow_pickle=True)
        else:
            self.indexes_items_in_city = \
                np.array([np.where(self.item_node == city_i)[0] for city_i in range(self.dim)])
            np.save(index_cache_name, self.indexes_items_in_city, allow_pickle=True) # we must
            # allow pickle due to nested arrays

        # if a city had less than max items they were pad with -1, this is only for the @njit
        # functions which do not take arrays of objects so our array must be of
        # type int not object. DO NOT USE FOR OTHER STUFF
        self.city_item_index_opt_do_not_use = np.array(
            list(itertools.zip_longest(*self.indexes_items_in_city, fillvalue=-1))).T


        #The graphs should be the same for all files with the same first part of the name, right?
        dist_cache_name = 'pickles/'+file.split('/')[1].split('_')[0]+'_dist'
        if os.path.isfile(dist_cache_name+'.npy'):
            print('Loading', dist_cache_name, 'from Cache')
            self.dist_cache = np.load(dist_cache_name+'.npy')
        else:
            self.dist_cache = squareform(np.ceil(pdist(self.node_coord)).astype(np.int32,
                                                                                copy=False))
        np.save(dist_cache_name, self.dist_cache)
        self.normalizing_constant = (self.max_speed - self.min_speed) / self.knapsack_capacity
        self.ttp_opt = TTP_OPT(self.dim, self.item_num, self.knapsack_capacity, self.min_speed,
                               self.max_speed, self.normalizing_constant, self.renting_ratio,
                               self.item_weight,self.item_profit,self.item_node)




SPEC = [
    ('dim', nb.int64),
    ('item_num', nb.int64),
    ('knapsack_capacity', nb.int64),
    ('min_speed', nb.float64),
    ('max_speed', nb.float64),
    ('normalizing_constant', nb.float64),
    ('renting_ratio', nb.float64),
    ('item_weight', nb.int64[:]),
    ('item_profit', nb.int64[:]),
    ('item_node', nb.int64[:])

]
# see http://numba.pydata.org/numba-doc/latest/user/jitclass.html
@jitclass(SPEC)
class TTP_OPT:
    def __init__(self, dim, item_num, knapsack_capacity,
                 min_speed, max_speed, normalizing_constant, renting_ratio, item_weight,
                 item_profit, item_node):
        self.dim = dim
        self.item_num = item_num
        self.knapsack_capacity = knapsack_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.normalizing_constant = normalizing_constant
        self.renting_ratio = renting_ratio
        self.item_weight = item_weight
        self.item_profit = item_profit
        self.item_node = item_node

