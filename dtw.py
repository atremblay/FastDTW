import logging
import numpy as np
import abc
from abc import ABCMeta
from pandas import Series


class Coord(object):
    """docstring for Coord"""
    def __init__(self, x, y):
        super(Coord, self).__init__()
        self.x = x
        self.y = y


class Cell(object):
    @profile
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        string = "({}, {})".format(self.x, self.y)
        return string

    @property
    def coord(self):
        return Coord(self.x, self.y)

    @property
    def matrix(self):
        return Coord(self.x + 1, self.y + 1)



class Left(Cell):
    """docstring for Left"""
    @profile
    def __init__(self, cell):
        self.x = cell.x - 1
        self.y = cell.y


class Up(Cell):
    """docstring for Up"""
    @profile
    def __init__(self, cell):
        self.x = cell.x
        self.y = cell.y - 1


class Diagonal(Cell):
    """docstring for Diagonal"""
    @profile
    def __init__(self, cell):
        self.x = cell.x - 1
        self.y = cell.y - 1


def optimal_warping_path(n, m, N, M, optimal_path, step=0):
    if n == 1:
        p = (1, m - 1)
        optimal_path[step] = p

    elif m == 1:
        p = (n - 1, 1)
        optimal_path[step] = p


def dtw(X, Y):
    pass

@profile
def distance_matrix(cost_matrix):
    dist_matrix = np.ones(cost_matrix.shape) * -1
    _distance(cost_matrix, dist_matrix)
    return dist_matrix

recursions = {}
@profile
def _distance(cost_matrix, dist_matrix, recursion = 0):
    # if recursion > 25:
    #     return

    # print(recursion)
    # print(dist_matrix)

    # print("cost matrix \n{}".format(cost_matrix.T))
    # print("shape {}".format(cost_matrix.shape))
    (x, y) = cost_matrix.shape

    cell = Cell(x-1, y-1)
    current_dist = dist_matrix[cell.coord.x, cell.coord.y]
    if current_dist != -1:
        return current_dist

    if recursion not in recursions:
        recursions[recursion] = []
    recursions[recursion].append(cell)

    left = Left(cell)
    up = Up(cell)
    diag = Diagonal(cell)
    # print("cell {}".format(cell))
    # print("left {}".format(left))
    # print("up {}".format(up))
    # print("diag {}".format(diag))

    cost = cost_matrix[cell.coord.x, cell.coord.y]

    # print("cost({}) = {}".format(cell, cost))

    if cell.coord.x == 0 and cell.coord.y == 0:
        dist_matrix[cell.coord.x, cell.coord.y] = cost
        return cost

    if cell.coord.x == 0:
        # print("Hit the left wall")
        d = _distance(
            cost_matrix[:1,:max(0, up.matrix.y)],
            dist_matrix, recursion + 1) + cost
        dist_matrix[cell.x,cell.y] = d
        return d

    if cell.coord.y == 0:
        # print("Hit the ceiling")
        dist_matrix[cell.x, cell.y] = _distance(
            cost_matrix[:max(0, left.matrix.x),:1],
            dist_matrix, recursion+1) + cost
        return dist_matrix[cell.x,cell.y]


    # print("{}Calculating up cell".format('-'*recursion))
    u = _distance(cost_matrix[:up.matrix.x,:up.matrix.y], dist_matrix, recursion+1),
    # print("{}Calculating left cell".format('-'*recursion))
    l= _distance(cost_matrix[:left.matrix.x,:left.matrix.y], dist_matrix, recursion+1),
    # print("{}Calculating diag cell".format('-'*recursion))
    r = _distance(cost_matrix[:diag.matrix.x,:diag.matrix.y], dist_matrix, recursion+1)
    min_dist = min(
        u, l, r
        )
    dist_matrix[cell.coord.x,cell.coord.y] = min_dist + cost
    return dist_matrix[cell.coord.x, cell.coord.y]

@profile
def cost_matrix(X, Y):
    # matrix = np.zeros((len(X), len(Y)))
    m = [cost(x, y) for x in X for y in Y]

    return np.array(m).reshape(len(X), len(Y))

@profile
def cost(x, y, *, kind='manhattan'):
    if kind == 'manhattan':
        return manhattan(x, y)
    else:
        raise AttributeError("Cost distance not implemented")



def manhattan(x, y):
    return abs(x - y)

x = np.arange(3)
y = np.arange(2)
c = cost_matrix(x, y)
print(x)
print(y)
print(c.T)
print(distance_matrix(c).T)

# longest_path = recursions[max(recursions.keys())]
# print(longest_path)
# print(recursions)