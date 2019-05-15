from model import TTSP

class NeighrestNeighbors:
    def __init__(self, ttsp:TTSP):
        self.ttsp = ttsp

    def optimize(self):
        permutation = [0]
        visited = [False]*self.ttsp.dim
        found = 1
        current = 0

        while found != self.ttsp.dim:
            min_dist = 0xFFFFFFFFEEE
            min_ele = 0
            visited[current] = True
            for i in range(self.ttsp.dim):
                if visited[i]:
                    continue
                if self.ttsp.dist(current, i) < min_dist:
                    min_ele = i
                    min_dist = self.ttsp.dist(current, i)
            current = min_ele
            found += 1
            permutation.append(min_ele)
        print(len(permutation))
        return permutation
