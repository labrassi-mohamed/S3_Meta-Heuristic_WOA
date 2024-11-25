from read import read_tai12a
from woa import WhaleOptimizationQAP
from animate_optimization import animate_optimization

flow_matrix, distance_matrix = read_tai12a("./tai12a.dat")

woaqap = WhaleOptimizationQAP(flow_matrix, distance_matrix, n_whales=50, max_iter=100)

# Original Algorithm
best_solution, best_fitness, history, positions_history= woaqap.optimize()

print(f"Best solution found Without local search: {best_solution}")
print(f"Best fitness value Without local search: {best_fitness}")

animate_optimization(positions_history, bounds=(-12, 12), optimum=best_solution)


# Impovement with local search
best_solution, best_fitness, history, positions_history= woaqap.optimize_with_local_search()

print(f"Best solution found With local search: {best_solution}")
print(f"Best fitness value With local search: {best_fitness}")

animate_optimization(positions_history, bounds=(-12, 12), optimum=best_solution)