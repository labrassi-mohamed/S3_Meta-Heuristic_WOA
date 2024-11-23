from read import read_tai12a
from woa import WhaleOptimizationQAP

flow_matrix, distance_matrix = read_tai12a("./tai12a.dat")

woaqap = WhaleOptimizationQAP(flow_matrix, distance_matrix, n_whales=5, max_iter=100)

best_solution, best_fitness, history = woaqap.optimize()

print(f"Best solution found: {best_solution}")
print(f"Best fitness value: {best_fitness}")
# print(f"Optimization history: {history}")
