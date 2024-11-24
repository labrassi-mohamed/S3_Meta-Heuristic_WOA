from read import read_tai12a
from woa import WhaleOptimizationQAP
from animate_optimization import animate_optimization

flow_matrix, distance_matrix = read_tai12a("./tai12a.dat")

woaqap = WhaleOptimizationQAP(flow_matrix, distance_matrix, n_whales=20, max_iter=100)

best_solution, best_fitness, history, positions_history= woaqap.optimize()

print(f"Best solution found: {best_solution}")
print(f"Best fitness value: {best_fitness}")

if True:
    animate_optimization(positions_history, bounds=(-12, 12), optimum=best_solution)
