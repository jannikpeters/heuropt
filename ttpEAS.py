from TestCase import TestCase
import ea_utils
from model import TTSP
import numpy as np
from evaluation_function import profit as calculate_profit
from numba import njit
from scipy.spatial import KDTree
from evaluation_function import t_opt
import warnings


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

        self.tree = KDTree(self.ttsp.node_coord)

        # init kp profit

        self.value = np.multiply(kp, ttspModel.item_profit).sum()
        self.weight = np.multiply(kp, ttspModel.item_weight).sum()

        # init rent
        self.city_weights = np.zeros(self.tour_size)  # weight at city i
        self.init_rent()

        self.rent = rent

        self.node_pos_in_tour = np.zeros(self.tour.shape, dtype=np.int)
        for i in range(len(self.tour)):
            node = self.tour[i]
            self.node_pos_in_tour[node] = i

        self.MUTATORS_defaults = {
            'kp_swap_item': (self.mutate_item_swap, 0.25),
            'kp_flip_item': (self.mutate_kp_flip, 0.125),
            'tour_city_swap': (self.mutate_city_swap, 0.25),
            'tour_reverse_subpath_neighbour': (self.mutate_reverse_subpath_neighbour, 0.375),
            'tour_reverse_subpath_random': (self.mutate_reverse_subpath_random, 0),
        }

        self.mutators_names = list(self.MUTATORS_defaults.keys())
        self.mutator_funcs = [self.MUTATORS_defaults[n][0] for n in self.mutators_names]

        self.rs = np.random.RandomState(seed)

    def init_rent(self):
        tour_size = self.tour_size
        for i in range(tour_size):
            city_i = self.tour[i % tour_size]

            self.city_weights[city_i] = self.added_weight_at_opt(city_i, self.kp,
                                                                 self.ttsp.item_weight,
                                                                 self.ttsp.city_item_index_opt_do_not_use)

    def _induce_profit_kp_change(self, kp: np.ndarray, kp_change_list: list):
        return ea_utils._induced_profit_kp_changes_opt(self.value,
                                                       self.weight,
                                                       kp,
                                                       kp_change_list,
                                                       self.tour_size,
                                                       self.tour,
                                                       self.city_weights,
                                                       self.ttsp.dist_cache,
                                                       self.ttsp.ttp_opt)

    def _induce_profit_swap_change(self, tour_city_swaps: np.ndarray):
        return ea_utils._induce_profit_swap_change_opt(self.value, tour_city_swaps,
                                                       self.tour_size, self.tour,
                                                       self.city_weights, self.ttsp.dist_cache
                                                       , self.ttsp.ttp_opt)

    def _induce_reverse_nearest_neighbour(self, origins, k=5):
        # TODO[Anton]: Might be hard to optimize due to the usage of the kdtree. Perhaps we can numba
        # TODO[Freya]: does not take too long anyways? the function that it calls is more relevant
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
            if max_abs < dif:  # enforce, that a node on the tour following is used
                max_abs = dif
                best_neighbor_node = i

        best_neighbor_node_pos_in_tour = self.node_pos_in_tour[best_neighbor_node]

        if best_neighbor_node == -1 or best_neighbor_node_pos_in_tour == node_pos_in + 1 or node_pos_in in [
            0, 1, 2, self.tour_size - 1, self.tour_size - 2, self.tour_size - 3]:  # just skip
            return None, None, None
        else:
            return self._induce_reverse_subpath(node_pos_in,
                                                best_neighbor_node_pos_in_tour), node_pos_in, best_neighbor_node_pos_in_tour

    def _induce_reverse_subpath(self, first_node, second_node):
        """
        Induce the profit of the changed tour: [:first_node]reversed([first_node:second_node+1])[second_node+1:]

        second_node > first_node + 1
        :param first_node: position of the first node in the tour
        :param second_node: position of the second node in the tour
        :return:
        """
        return ea_utils._induce_reverse_subpath_opt(first_node, second_node,
                                                    self.tour, self.tour_size,
                                                    self.city_weights, self.ttsp.ttp_opt,
                                                    self.value, self.ttsp.dist_cache)

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

    def _commit_reversal(self, node_pos_in, best_neighbor_node_pos_in_tour):
        assert node_pos_in < best_neighbor_node_pos_in_tour
        start, reversal, end = np.split(self.tour,
                                        [node_pos_in, best_neighbor_node_pos_in_tour + 1])
        assert reversal[0] == self.tour[node_pos_in]
        assert reversal[-1] == self.tour[best_neighbor_node_pos_in_tour]
        reversal = np.flip(reversal, 0)
        self.tour = np.concatenate([start, reversal, end])

    def mutate_kp_flip(self, current_profit):
        kp_changes = self.kp_selection_func(self.ttsp.item_num)
        new_value, new_weight, new_profit = self._induce_profit_kp_change(self.kp, kp_changes)

        if new_profit >= current_profit and new_weight <= self.ttsp.knapsack_capacity:
            self.value = new_value
            self.weight = new_weight
            self._commit_kp_changes(self.kp, kp_changes)
            return current_profit

        return None

    def mutate_city_swap(self, current_profit):

        number_of_changes = self.rs.binomial(n=self.tour_size,
                                             p=3 / self.tour_size) + 1  # p= ??
        neighbor_swaps = self.rs.choice(self.tour_size, number_of_changes, replace=False)

        new_profit = self._induce_profit_swap_change(neighbor_swaps)
        if new_profit >= current_profit:
            self._commit_city_swaps(neighbor_swaps)
            return new_profit

        return None

    def mutate_reverse_subpath_random(self, current_profit):
        a = self.rs.choice(self.tour_size, replace=False)
        b = self.rs.choice(self.tour_size, replace=False)
        first = min(a, b)
        second = max(a, b)

        if first != 0 and second != self.tour_size and first + 1 < second:

            new_profit = self._induce_reverse_subpath(first, second)
            if new_profit is not None and new_profit >= current_profit:
                self._commit_reversal(first, second)
                return new_profit
        return None

    def mutate_reverse_subpath_neighbour(self, current_profit):
        reverse_origin = self.rs.choice(self.tour_size, 1, replace=False)
        new_profit, node_pos_in, best_neighbor_node_pos_in_tour = self._induce_reverse_nearest_neighbour(
            reverse_origin)
        if new_profit is not None and new_profit >= current_profit:
            self._commit_reversal(node_pos_in, best_neighbor_node_pos_in_tour)
            return new_profit
        return None

    def mutate_item_swap(self, current_profit):
        first = -1
        second = -1
        for i in range(10):
            first = self.rs.choice(self.ttsp.item_num, 1)[0]
            if self.kp[first] == 0:
                break
        for i in range(10):
            second = self.rs.choice(self.ttsp.item_num, 1)[0]
            if self.kp[second] == 1:
                break
        if first != -1 and second != -1:
            kp_changes = [first, second]
            new_value, new_weight, new_profit = self._induce_profit_kp_change(self.kp,
                                                                              kp_changes)
            if new_profit >= current_profit and new_weight <= self.ttsp.knapsack_capacity:
                self.value = new_value
                self.weight = new_weight
                self._commit_kp_changes(self.kp, kp_changes)
                return new_profit

        return None

    def optimize(self, mutators: dict = None):
        """

        :param mutators: Dict of 'mutator_name' : probabilty of occurrence
        :return: optimized tour
        """

        if mutators is None:
            # janniks standard config
            mutator_names = self.mutators_names
            mutator_probs = np.cumsum([self.MUTATORS_defaults[name][1] for name in mutator_names])
        else:
            mutator_names = list(mutators.keys())
            mutator_probs = np.cumsum([mutators[name] for name in mutator_names])

        assert mutator_probs[-1] == 1, "Probabilities should add up to 1."
        mutator_funcs = [self.MUTATORS_defaults[name][0] for name in mutator_names]

        profit = calculate_profit(self.tour, self.kp, self.ttsp)

        while not self.stopping_criterion.is_done(profit):
            mutator = np.argmax(mutator_probs > self.rs.rand())

            new_profit = mutator_funcs[mutator](current_profit=profit)
            if new_profit is not None:
                profit = new_profit

        p1 = calculate_profit(self.tour, self.kp, self.ttsp)
        assert p1 == profit, 'Somethings wrong with the mutators!'

        return profit, self.kp, self.tour, self.stopping_criterion
