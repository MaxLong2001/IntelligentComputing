# -*- coding: utf-8 -*-
import time

import numpy as np

n = 0
dist_matrix = 0
flow_matrix = 0
generation_max = 0
population = 50
w = 1
c1 = 0.75
c2 = 0.75
best_group = float('inf')
best_cost = float('inf')
particle_dict = {}


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
    generation_max = n * 200
    print(n, dist_matrix, flow_matrix, generation_max)


def cost(group):
    return sum(np.sum(flow_matrix * dist_matrix[group[:, None], group], 1))


def update_v(a, b):
    ans = []
    v = np.eye(len(a))
    for i in range(len(a)):
        a_idx = np.argwhere(a == i)
        b_idx = np.argwhere(b == i)
        tmp = np.eye(len(a))
        tmp[[a_idx, b_idx], :] = tmp[[b_idx, a_idx], :]
        if np.any(tmp != np.eye(len(a))):
            ans.append(tmp)
        v = np.dot(v, tmp)
        a = np.dot(a, tmp)
    return ans


def move():
    global best_cost, best_group
    for i in range(population):
        v_0 = particle_dict[i]["velocity"]
        v_self = update_v(particle_dict[i]["group"], particle_dict[i]["best"])
        v_group = update_v(particle_dict[i]["group"], best_group)

        r1 = np.random.random(size=[len(v_self), 1])
        r2 = np.random.random(size=[len(v_group), 1])

        v_1 = np.eye(n)
        for j in range(len(v_self)):
            if r1[j] <= c1:
                v_1 = np.dot(v_self[j], v_1)
        v_2 = np.eye(n)
        for j in range(len(v_group)):
            if r2[j] <= c2:
                v_2 = np.dot(v_group[j], v_2)

        v_0 = v_0 * w
        v = np.dot(v_0, np.dot(v_1, v_2))
        particle_dict[i]["velocity"] = v

        particle_dict[i]["group"] = np.dot(v, particle_dict[i]["group"])
        particle_dict[i]["group"] = np.array([int(j) for j in particle_dict[i]["group"]])

        tmp_cost = cost(particle_dict[i]["group"])
        if tmp_cost < particle_dict[i]["best_cost"]:
            particle_dict[i]["best_cost"] = tmp_cost
            particle_dict[i]["best"] = particle_dict[i]["group"]
        if tmp_cost < best_cost:
            best_group = particle_dict[i]["group"]
            best_cost = tmp_cost


def pso_run():
    global best_cost, best_group, particle_dict
    best_group = float('inf')
    best_cost = float('inf')
    particle_dict = {}
    for i in range(population):
        group = np.random.permutation(n)
        particle_dict[i] = {"group": group, "cost": cost(group), "velocity": np.eye(n, dtype=int),
                            "best": group, "best_cost": cost(group)}
        if particle_dict[i]["cost"] < best_cost:
            best_group = particle_dict[i]["group"]
            best_cost = particle_dict[i]["cost"]

    for i in range(generation_max):
        move()
        print("???{}???????????????????????????{}???????????????{}".format(i, best_group, best_cost))

    return best_group, best_cost


if __name__ == '__main__':
    data_num = 32
    read_data(f'.\qap-problems\QAP{data_num}.dat')
    with open(f'.\qap-solutions\QAP{data_num}-PSO.txt', 'w', encoding='utf-8') as f:
        for i in range(10):
            time_start = time.perf_counter()
            res_group, res_cost = pso_run()
            time_end = time.perf_counter()
            f.write(f'No {i + 1}:\n')
            f.write(f'result:\t{res_group}\ncost:\t{res_cost}\ntime:\t{time_end - time_start}\n')
