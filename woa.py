import numpy as np
import random
from typing import Tuple, List

class WhaleOptimizationQAP:
    def __init__(self, flow_matrix: np.ndarray, distance_matrix: np.ndarray, 
                 n_whales: int = 50, max_iter: int = 100) -> None:
        
        """
            Initialize WOA for QAP
            
            Args:
                flow_matrix: Matrix representing flow between facilities
                distance_matrix: Matrix representing distances between locations
                n_whales: Number of search agents (whales)
                max_iter: Maximum number of iterations
                n: Number of facilities or Distances
        """
        self.flow = flow_matrix
        self.distance = distance_matrix
        self.n = len(flow_matrix)
        self.n_whales = n_whales
        self.max_iter = max_iter
        
    def __calculate_fitness(self, solution: np.ndarray) -> float:
        """Calculate the fitness (total cost) of a solution."""
        cost = 0
        for i in range(self.n):
            for j in range(self.n):
                cost += self.flow[i][j] * self.distance[solution[i]][solution[j]]
        return cost
    
    def __create_random_solution(self) -> np.ndarray:
        """Create a random permutation solution"""
        return np.random.permutation(self.n)
    
    def __encircling_prey(self, current_pos: np.ndarray, best_pos: np.ndarray, A: float, C: float) -> np.ndarray:
        D = abs(C * best_pos - current_pos)
        new_pos = best_pos - A * D
        return np.argsort(new_pos)
    
    def __search_for_prey(self, current_pos: np.ndarray, random_pos: np.ndarray, A: float, C: float) -> np.ndarray:
        D = abs(C * random_pos - current_pos)
        new_pos = random_pos - A * D
        return np.argsort(new_pos) 
    
    def __bubble_net_attack(self, current_pos: np.ndarray, best_pos: np.ndarray, l: float) -> np.ndarray:
        D = abs(best_pos - current_pos)
        b = 1
        new_pos = D * np.exp(l * b) * np.cos(2 * np.pi * l) + best_pos
        return np.argsort(new_pos)
    
    def __amend_position(self, position: np.ndarray) -> np.ndarray:
        """Ensure position is a valid permutation."""
        return np.argsort(position)
   
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        This method it's the Whale Optimization Algorithm.
        
        Returns:
            Tuple containing:
            - Best solution found (permutation)
            - Best fitness found
            - History of best fitness values
        """
        
        # Initialize the whales population Xi (i = 1, 2, ..., n)
        population = [self.__create_random_solution() for _ in range(self.n_whales)]
        
        # Calculate the fitness of each search agent
        fitness_values = [self.__calculate_fitness(pos) for pos in population]
        positions_history = [population.copy()]
        
        # X* = the best search agent
        best_idx = np.argmin(fitness_values)
        best_pos = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        fitness_history = [best_fitness]
        t = 0
        
        # Main loop
        while t < self.max_iter:
            # For each search agent
            for i in range(self.n_whales):
                # Update a, A, C, l, and p
                a = 2 - t * (2 / self.max_iter)  # a decreases linearly from 2 to 0
                r = random.random()
                A = 2 * a * r - a  # Eq.
                C = 2 * r  # Eq. 
                l = random.uniform(-1, 1)  # parameter for spiral
                p = random.random()  # probability for movement type
            
                if p < 0.5:
                    if abs(A) < 1:
                        # Update position by (Encircling prey)
                        new_pos = self.__encircling_prey(population[i], best_pos, A, C)
                    else:
                        # Select a random search agent
                        rand_idx = random.randint(0, self.n_whales-1)
                        random_pos = population[rand_idx]
                        # Update position by (Searching for prey)
                        new_pos = self.__search_for_prey(population[i], random_pos, A, C)
                else:
                    # Update position by (Bubble-net attacking)
                    new_pos = self.__bubble_net_attack(population[i], best_pos, l)
                
                # Amend position if needed
                new_pos = self.__amend_position(new_pos)
                population[i] = new_pos
                
                # Calculate fitness
                new_fitness = self.__calculate_fitness(new_pos)
                fitness_values[i] = new_fitness
                
                # Update X* if there is a better solution
                if new_fitness < best_fitness:
                    best_pos = new_pos.copy()
                    best_fitness = new_fitness
            
            fitness_history.append(best_fitness)
            positions_history.append(population.copy())
            t += 1
        
        return best_pos, best_fitness, fitness_history, positions_history
        
        
        