# -*- coding: utf-8 -*-
import numpy as np
from math import *

n = 0
dist_matrix = 0
flow_matrix = 0
generation_max = 0
population = 100
p_mutation = 0.02
p_cross = 0.5


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


def cross(parent_1, parent_2, cross_idx1, cross_idx2):
    if cross_idx1 > cross_idx2:
        cross_idx1, cross_idx2 = cross_idx2, cross_idx1

    idx = list(range(cross_idx2 + 1, n)) + list(range(0, cross_idx1))
    child_1, child_2 = parent_1.copy(), parent_2.copy()

    child_1["chromosome"][idx] = [p for p in parent_2["chromosome"] if
                                  p not in parent_1["chromosome"][cross_idx1:cross_idx2 + 1]]
    child_2["chromosome"][idx] = [p for p in parent_1["chromosome"] if
                                  p not in parent_2["chromosome"][cross_idx1:cross_idx2 + 1]]
    child_1["cost"], child_2["cost"] = cost(child_1["chromosome"]), cost(child_2["chromosome"])

    return child_1, child_2


def mutation(individual, mutation_idx1, mutation_idx2):
    if mutation_idx1 > mutation_idx2:
        mutation_idx1, mutation_idx2 = mutation_idx2, mutation_idx1
    individual["chromosome"][mutation_idx1], individual["chromosome"][mutation_idx2] = individual["chromosome"][
                                                                                           mutation_idx2], \
                                                                                       individual["chromosome"][
                                                                                           mutation_idx1]
    individual["cost"] = cost(individual["chromosome"])


def mutation_cost(cur_chromosome, cur_cost, cost_idx1, cost_idx2):
    new_chromosome = np.copy(cur_chromosome)
    new_chromosome[cost_idx1], new_chromosome[cost_idx2] = new_chromosome[cost_idx2], new_chromosome[cost_idx1]

    mutation_cost_res = cur_cost - sum(np.sum(flow_matrix[[cost_idx1, cost_idx2], :] * dist_matrix[
        cur_chromosome[[cost_idx1, cost_idx2], None], cur_chromosome], 1))
    mutation_cost_res += sum(np.sum(flow_matrix[[cost_idx1, cost_idx2], :] * dist_matrix[
        new_chromosome[[cost_idx1, cost_idx2], None], new_chromosome], 1))

    idx = list(range(n))
    del (idx[cost_idx1])
    del (idx[cost_idx2 - 1])

    mutation_cost_res -= sum(sum(flow_matrix[idx][:, [cost_idx1, cost_idx2]] * dist_matrix[
        cur_chromosome[idx, None], cur_chromosome[[cost_idx1, cost_idx2]]]))
    mutation_cost_res += sum(
        sum(flow_matrix[idx][:, [cost_idx1, cost_idx2]] * dist_matrix[
            new_chromosome[idx, None], new_chromosome[[cost_idx1, cost_idx2]]]))

    return mutation_cost_res


def cost(chromosome):
    return sum(np.sum(flow_matrix * dist_matrix[chromosome[:, None], chromosome], 1))


def ga_run():
    solutions = np.zeros(generation_max, dtype=np.int64)
    num_cross = ceil(population / 2.0 * p_cross)
    num_mutation = ceil(population * n * p_mutation)
    data_type = np.dtype([('chromosome', str(n) + 'int'), ('cost', np.int64)])
    # print (num_crosses)       #20
    # print (num_mutations)     #12

    # Generate initial population
    parents = np.zeros(population, dtype=data_type)

    for individual_solution in parents:
        individual_solution["chromosome"] = np.random.permutation(n)
        individual_solution["cost"] = cost(individual_solution["chromosome"])

    parents.sort(order="cost", kind='mergesort')
    print(parents)
    for generation in range(generation_max):
        # Tournament selection
        contestant_idx = np.empty(population, dtype=np.int32)
        for index in range(population):
            contestant_idx[index] = np.random.randint(low=0, high=np.random.randint(1, population))
        contestant_pairs = zip(contestant_idx[0:2 * num_cross:2], contestant_idx[1:2 * num_cross:2])
        # print(contestant_pairs)

        # Crossover the selected chromosomes
        children = np.zeros(population, dtype=data_type)
        cross_points = np.random.randint(n, size=2 * num_cross)

        for pairs, idx1, idx2 in zip(contestant_pairs, range(0, 2 * num_cross, 2), range(1, 2 * num_cross, 2)):
            children[idx1], children[idx2] = cross(parents[pairs[0]], parents[pairs[1]], cross_points[idx1],
                                                   cross_points[idx2])
            # print (index1)
        children[2 * num_cross:] = parents[contestant_idx[2 * num_cross:]].copy()

        # Mutate the children
        mutant_children_Idx = np.random.randint(population, size=num_mutation)
        mutant_allele = np.random.randint(n, size=2 * num_mutation)
        for index, gen_Index in zip(mutant_children_Idx, range(num_mutation)):
            mutation(children[index], mutant_allele[2 * gen_Index], mutant_allele[2 * gen_Index + 1])

        # Replace the parents with children
        children.sort(order="cost", kind='mergesort')

        # 1. with elitism -- it gets to the optimal faster
        if children["cost"][0] > parents["cost"][0]:
            children[-1] = parents[0]
        # 2. replace without elitism
        parents = children  # replace the parent with children

        parents.sort(order="cost", kind='mergesort')
        print("第{}次迭代，最优位置为{}，最优值为{}".format(generation, parents[0]["chromosome"], parents[0]["cost"]))
        solutions[generation] = parents[0]["cost"]

    return parents[0]["chromosome"], parents[0]["cost"]


if __name__ == "__main__":
    read_data('.\qap-problems\QAP12.dat')
    chromosome_res, cost_res = ga_run()
