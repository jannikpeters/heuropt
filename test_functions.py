import numpy as np


class TestFunction:
    name = None

    def __init__(self, parameters: dict = None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def optimum(self, n: int):
        raise NotImplementedError("Subclass should implement.")

    def f(self, bitstring: np.ndarray):
        raise NotImplementedError("Subclass should implement.")


class OneMax(TestFunction):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)
        self.name = 'OneMax'

    def optimum(self, n: int):
        return n

    def f(self, bitstring: np.ndarray):
        return bitstring.sum()


class LeadingOnes(TestFunction):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)
        self.name = 'LeadingOnes'

    def optimum(self, n: int):
        return n

    def f(self, bitstring: np.ndarray):
        if 0 not in bitstring:
            return bitstring.sum()
        return bitstring[0:np.where(bitstring == 0)[0][0]].sum()


class Jump(TestFunction):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)
        self.name = 'Jump(%s)' % self.parameters['k']

    def optimum(self, n: int):
        return n

    def f(self, bitstring: np.ndarray):
        aggregate = sum(bitstring)
        if aggregate < len(bitstring) - self.parameters['k']:
            return aggregate
        if aggregate < len(bitstring):
            return len(bitstring) - self.parameters['k']
        return aggregate


class BinVal(TestFunction):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)
        self.name = 'BinVal'

    def optimum(self, n: int):
        return (2 ** n) - 1

    def f(self, bitstring: np.ndarray):
        """ Convert binary string to real valued number"""
        s = ''
        for val in bitstring:
            s += str(int(val))
        return int(s, 2)


class RoyalRoads(TestFunction):

    def __init__(self, parameters: dict = None):
        super().__init__(parameters)
        self.name = 'RoyalRoads(%s)' % self.parameters['r']

    def optimum(self, n: int):
        assert n % self.parameters['r'] == 0, "N should be divisible by r by definition."
        return n

    def f(self, bitstring: np.ndarray):
        """ Number of groups made up of only ones, where groups are created by intersecting at consecutive values. """
        r = self.parameters['r']
        royal_roads = 0
        n = len(bitstring)
        for group in range(int(n / r)):
            royal = True
            for val in bitstring[group * r:group * (r + 1)]:
                if val == 0:
                    royal = False
                    break

            if royal:
                royal_roads += 1

        return royal_roads * r
