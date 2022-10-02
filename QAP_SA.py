# -*- coding: utf-8 -*-
import numpy as np
from math import *

n = 0
dist_matrix = 0
flow_matrix = 0
generation_max = 0
cooling_rate = 0.05
temp = 1000.0


def read_data(file_dir):
    global n, dist_matrix, flow_matrix, generation_max, population
    file = open(file_dir, 'r')
    n = int(file.readline())
    distances = []
    flows = []
    lines = [line for line in file.readlines() if line.strip()]
    for i in range(n):
        distances.append(list(map(int, lines[i].split())))
    for i in range(n):
        flows.append(list(map(int, lines[i + n].split())))
    dist_matrix = np.array(distances)
    flow_matrix = np.array(flows)
    generation_max = n * 1000
    print(n, dist_matrix, flow_matrix, generation_max)


def cost(group):
    return sum(np.sum(flow_matrix * dist_matrix[group[:, None], group], 1))


def acceptance_probability(old_cost, new_cost, T):
    if new_cost < old_cost:
        return 1.0
    else:
        return np.exp((old_cost - new_cost) / T)


def sa_run():
    global temp
    solution = np.random.permutation(n)
    it = 0
    while temp > 0.05:
        for i in range(generation_max):
            new_solution = np.copy(solution)
            idx1, idx2 = np.random.randint(0, n, 2)
            new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
            ap = acceptance_probability(cost(solution), cost(new_solution), temp)
            if ap > np.random.rand():
                solution = new_solution
        temp *= 1 - cooling_rate
        it += 1
        print("第{}次迭代，最优位置为{}，最优值为{}".format(it, solution, cost(solution)))


if __name__ == '__main__':
    read_data('.\qap-problems\QAP12.dat')
    sa_run()
