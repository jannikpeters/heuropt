from TestCase import TestCase
from model import TTSP
import numpy as np
from evaluation_function import profit as calculate_profit
from numba import njit
from scipy.spatial import KDTree


class OnePlusOneEA():

    def __init__(self,
                 ttspModel: TTSP,
                 tour: np.ndarray,
                 kp: np.ndarray,
                 stopping_criterion: TestCase,
                 kp_selection_func,
                 rent,
                 seed):

        self.ttsp = ttspModel
        self.stopping_criterion = stopping_criterion
        self.kp_selection_func = kp_selection_func
        self.tour = tour
        self.kp = kp
        self.tour_size = len(tour)
        self.rent = rent

        # init kp profit

        self.value = np.multiply(kp, ttspModel.item_profit).sum()
        self.weight = np.multiply(kp, ttspModel.item_weight).sum()

        # init rent
        self.city_weights = np.zeros(self.tour_size)  # weight at city i
        self.init_rent()

        self.rent = rent

    def init_rent(self):
        tour_size = self.tour_size
        for i in range(tour_size):
            city_i = self.tour[i % tour_size]

            self.city_weights[city_i] = self.added_weight_at_opt(city_i, self.kp, self.ttsp.item_weight,
                                                                 self.ttsp.city_item_index_opt_do_not_use)

    def _induce_profit_kp_change(self, kp: np.ndarray, kp_change_list: list):

        # apply knapsack changes
        new_value = self.value
        new_weight = self.weight
        cities = []

        for item_index in kp_change_list:
            # if the item is currently 0 we will make it 1 and vice versa
            new_value += ((-2) * kp[item_index] + 1) * self.ttsp.item_profit[item_index]
            new_weight += ((-2) * kp[item_index] + 1) * self.ttsp.item_weight[item_index]
            cities.append(self.ttsp.item_node[item_index])

        rent = 0
        current_weight = 0

        for i in range(self.tour_size):
            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i + 1) % self.tour_size]

            current_weight += self.city_weights[city_i]
            weight_changes = 0
            for j, item_index in enumerate(kp_change_list):
                if cities[j] == city_i:
                    weight_changes += ((-2) * kp[item_index] + 1) * self.ttsp.item_weight[item_index]

            current_weight += weight_changes

            tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                             self.ttsp.normalizing_constant,
                             current_weight)
            rent += tij

        if new_weight > self.ttsp.knapsack_capacity:
            return -1, new_weight, -1
        else:
            return new_value, new_weight, new_value - self.ttsp.renting_ratio * rent

    def _induce_reverse_nearest_neighbour(self,origins,k=5):
        node_pos_in = origins[0]
        node = self.tour[node_pos_in]
        node_coord = self.ttsp.node_coord[node, :]
        tree = self.tree
        distances, idx = tree.query(node_coord, k=k)
        best_neighbor_node = -1
        max_abs = 0

        for i in idx:
            i_coord = tree.data[i]
            i_pos_in_tour = self.node_pos_in_tour[i]
            dif = (i_pos_in_tour - node_pos_in)
            if max_abs < dif: # enforce, that a node on the tour following is used
                max_abs = dif
                best_neighbor_node = i

        best_neighbor_node_pos_in_tour = self.node_pos_in_tour[best_neighbor_node]

        if best_neighbor_node == -1 or best_neighbor_node_pos_in_tour == node_pos_in + 1 or node_pos_in in [0,1,2,self.tour_size-1,self.tour_size-2,self.tour_size-3]: # just skip
            return None, None, None
        else:
            return self._induce_reverse_subpath(node, node_pos_in, best_neighbor_node, best_neighbor_node_pos_in_tour)


    def _induce_reverse_subpath(self,node, node_pos_in, best_neighbor_node, best_neighbor_node_pos_in_tour):
        """
        plan:
        origin -> bestneighbour+1
        bestneighbor <- origin-1
        (origin....bestneighbour)^T
        """

        assert node_pos_in < best_neighbor_node_pos_in_tour
        assert node_pos_in + 1 != best_neighbor_node_pos_in_tour

        start, reversal, end = np.split(self.tour, [node_pos_in, best_neighbor_node_pos_in_tour + 1])
        assert reversal[0] == self.tour[node_pos_in]
        assert reversal[-1] == self.tour[best_neighbor_node_pos_in_tour]
        reversal = np.flip(reversal,0)
        this_tour = np.concatenate([start, reversal, end])

        rent = 0
        current_weight = 0
        k = 0

        # distances btw [0,origin-1]
        for i in range(node_pos_in-1):
            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i + 1) % self.tour_size]
            current_weight += self.city_weights[city_i]
            tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                             self.ttsp.normalizing_constant,
                             current_weight)
            rent += tij
            assert this_tour[k] == city_i
            k += 1

        # new edge (origin-1->best_neighbour)
        city_i = self.tour[(node_pos_in-1) % self.tour_size]
        city_ip1 = self.tour[best_neighbor_node_pos_in_tour % self.tour_size]
        current_weight += self.city_weights[city_i]
        tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                         self.ttsp.normalizing_constant,
                         current_weight)
        rent += tij
        assert this_tour[k] == city_i
        k += 1

        # [best_neighbour -> origin] (reversed)
        for i in reversed(range(node_pos_in+1,best_neighbor_node_pos_in_tour+1)):

            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i - 1) % self.tour_size]
            current_weight += self.city_weights[city_i]
            tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                             self.ttsp.normalizing_constant,
                             current_weight)
            rent += tij
            assert this_tour[k] == city_i
            k += 1

        # new edge (origin -> best-neigbourp1)
        city_i = self.tour[node_pos_in % self.tour_size]
        city_ip1 = self.tour[(best_neighbor_node_pos_in_tour + 1) % self.tour_size]
        current_weight += self.city_weights[city_i]
        tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                         self.ttsp.normalizing_constant,
                         current_weight)
        rent += tij
        assert this_tour[k] == city_i
        k += 1

        # the rest
        for i in range(best_neighbor_node_pos_in_tour + 1,self.tour_size):
            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i + 1) % self.tour_size]
            current_weight += self.city_weights[city_i]
            tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                             self.ttsp.normalizing_constant,
                             current_weight)
            rent += tij
            assert this_tour[k] == city_i
            k += 1
        cost = self.value - self.ttsp.renting_ratio * rent
        assert calculate_profit(this_tour,self.kp, self.ttsp) == cost

        return cost, node_pos_in, best_neighbor_node_pos_in_tour

    def _induce_profit_swap_change(self, tour_city_swaps: list):
        " tour_city_swaps is a list denoting the cities swapping with their neighbours."

        # apply knapsack changes
        new_value = self.value
        new_weight = self.weight

        rent = 0
        current_weight = 0

        # the following tour_city_swaps are ignored:
        # cities where previous city was already selected for swap
        # 0 or n

        swap_phase1 = False
        swap_phase2 = False
        visited = np.zeros(self.tour_size)
        for i in range(self.tour_size):
            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i + 1) % self.tour_size]

            if swap_phase1:
                temp = city_i
                city_i = city_ip1
                city_ip1 = temp
                swap_phase1 = False
                swap_phase2 = True

            elif swap_phase2:
                city_i = self.tour[(i - 1) % self.tour_size]
                swap_phase2 = False

            elif (i != self.tour_size - 2 and i != self.tour_size - 1) and i in tour_city_swaps:
                city_ip1 = self.tour[(i + 2) % self.tour_size]
                swap_phase1 = True

            current_weight += self.city_weights[city_i]
            visited[city_i] = 1
            tij = self.t_opt(city_i, city_ip1, self.ttsp.dist_cache, self.ttsp.max_speed,
                             self.ttsp.normalizing_constant,
                             current_weight)
            rent += tij
        assert(visited.sum() == self.tour_size)
        return new_value - self.ttsp.renting_ratio * rent

    def t_opt(self, city_i: int, city_j: int, dist_matr: np.ndarray, max_speed: int, norm_const,
              current_weight: int):
        # Todo: if someone finds a way to make this faster, go ahead! It is the most called function
        return dist_matr[city_i, city_j] / (max_speed - current_weight * norm_const)

    def added_weight_at_opt(self, city_i: int, bit_string: np.ndarray, item_weight: np.ndarray,
                            index_city_items: np.ndarray):
        # if a city had less than max items they were padded with -1 we need to remove these
        city_items = index_city_items[city_i]
        valid_indexes = np.where(city_items != -1)
        unpadded_indexes = city_items[valid_indexes]
        if len(unpadded_indexes) == 0:
            return 0
        is_taken = bit_string[unpadded_indexes]
        weights = item_weight[unpadded_indexes]
        mult = np.multiply(is_taken, weights)
        res = mult.sum()
        return res

    def _commit_kp_changes(self, kp, change_list):
        for item in change_list:
            kp[item] = 1 - kp[item]
            if kp[item] == 0:
                self.city_weights[self.ttsp.item_node[item]] -= self.ttsp.item_weight[item]
            else:
                self.city_weights[self.ttsp.item_node[item]] += self.ttsp.item_weight[item]

        """
        i - 1->1
        i+1 - i   . 2->3
        i+2 - i+1  - i. 3->2
        i+3 - i +2 - i+1. 4->4
        a) b) c) 
        
        """

    def _commit_city_swaps(self, tour_city_swaps):

        swap_phase1 = False
        swap_phase2 = False

        for i in range(self.tour_size):
            city_i = self.tour[i % self.tour_size]
            city_ip1 = self.tour[(i + 1) % self.tour_size]

            if swap_phase1:
                swap_phase1 = False
                swap_phase2 = True

            elif swap_phase2:
                swap_phase2 = False


            elif (i != self.tour_size - 2 and i != self.tour_size - 1) and i in tour_city_swaps:
                tmp = city_ip1
                self.tour[(i + 1) % self.tour_size] = self.tour[(i + 2) % self.tour_size]
                self.tour[(i + 2) % self.tour_size] = tmp
                swap_phase1 = True

    def _commit_reversal(self,node_pos_in, best_neighbor_node_pos_in_tour):
        assert node_pos_in < best_neighbor_node_pos_in_tour
        start, reversal, end = np.split(self.tour,[node_pos_in,best_neighbor_node_pos_in_tour+1])
        assert reversal[0] == self.tour[node_pos_in]
        assert reversal[-1] == self.tour[best_neighbor_node_pos_in_tour]
        reversal = np.flip(reversal,0)
        self.tour = np.concatenate([start, reversal, end])


    def optimize(self):

        kp = self.kp
        n = self.ttsp.item_num

        self.tree = KDTree(self.ttsp.node_coord)
        self.node_pos_in_tour = np.zeros(self.tour.shape,dtype=np.int)
        for i in range(len(self.tour)):
            node = self.tour[i]
            self.node_pos_in_tour[node] = i

        knapsack_change = False
        cut_and_insert_tour = True
        tour_neig_change = False
        profit = calculate_profit(self.tour, self.kp, self.ttsp)

        while not self.stopping_criterion.is_done(self.value):
            knapsack_change = np.random.rand()
            if knapsack_change > 1:

                # knapsack changes
                kp_changes = self.kp_selection_func(n)
                new_value, new_weight, new_profit = self._induce_profit_kp_change(kp, kp_changes)

                if new_profit >= profit and new_weight <= self.ttsp.knapsack_capacity:
                    self.value = new_value
                    self.weight = new_weight
                    profit = new_profit
                    self._commit_kp_changes(kp, kp_changes)
                    print('k',profit)

            elif knapsack_change < 0:
                # tausche mit neighbour
                number_of_changes = np.random.binomial(n=self.tour_size, p=3 / self.tour_size) +1  # p= ??
                neighbor_swaps = np.random.choice(self.tour_size, number_of_changes, replace=False)
                #print(neighbor_swaps)
                new_profit = self._induce_profit_swap_change(neighbor_swaps)

                if new_profit >= profit:

                    profit = new_profit
                    print('tour',profit)
                    self._commit_city_swaps(neighbor_swaps)
                    #print(new_profit, calculate_profit(self.tour, self.kp, self.ttsp))

            elif knapsack_change > 0:

                reverse_origin = np.random.choice(self.tour_size, 1, replace=False)
                new_profit, node_pos_in, best_neighbor_node_pos_in_tour = self._induce_reverse_nearest_neighbour(reverse_origin)
                if new_profit is not None and new_profit/profit < 1.008:
                    profit = new_profit
                    #print('***%s' % self.stopping_criterion.steps)
                    self._commit_reversal(node_pos_in, best_neighbor_node_pos_in_tour)
                    print(profit)
                    #print(calculate_profit(self.tour, self.kp, self.ttsp))
            else:
                first = -1
                second = -1
                for i in range(10):
                    first = np.random.choice(n,1)[0]
                    if self.kp[first] == 0:
                        break
                for i in range(10):
                    second = np.random.choice(n,1)[0]
                    if self.kp[second] == 1:
                        break
                if first != -1 and second != -1:
                    kp_changes=[first, second]
                    new_value, new_weight, new_profit = self._induce_profit_kp_change(kp, kp_changes)
                    if new_profit >= profit and new_weight <= self.ttsp.knapsack_capacity:
                        self.value = new_value
                        self.weight = new_weight
                        profit = new_profit
                        self._commit_kp_changes(kp, kp_changes)
                        print('swap', profit)





        profit = calculate_profit(self.tour, self.kp, self.ttsp)

        return profit, self.kp, self.tour, self.stopping_criterion
