import numpy as np

def calculate_fitness_with_details(flow_matrix, distance_matrix, solution):
    """
    Calculate fitness with detailed breakdown of costs
    """
    n = len(solution)
    total_cost = 0
    details = []
    
    # For each pair of facilities
    for i in range(n):
        for j in range(n):
            # Get the flow between facilities i and j
            flow = flow_matrix[i][j]
            
            # Get the distance between their assigned locations
            dist = distance_matrix[solution[i]][solution[j]]
            
            # Calculate cost contribution
            cost = flow * dist
            
            if flow > 0:  # Only show non-zero flows
                details.append(
                    f"Facilities {i}-{j} (flow={flow}) assigned to locations "
                    f"{solution[i]}-{solution[j]} (distance={dist}): cost={cost}"
                )
            
            total_cost += cost
    
    return total_cost, details

# Example problem
flow_matrix = np.array([
    [0, 5, 2],
    [5, 0, 3],
    [2, 3, 0]
])

distance_matrix = np.array([
    [0, 4, 6],
    [4, 0, 2],
    [6, 2, 0]
])

# Example solution: Facility 0->2, Facility 1->0, Facility 2->1
solution = [2, 0, 1]

# Calculate fitness with details
total_cost, details = calculate_fitness_with_details(flow_matrix, distance_matrix, solution)

print("Detailed Cost Breakdown:")
for detail in details:
    print(detail)
print(f"\nTotal Cost: {total_cost}")

# Let's try different solutions to compare
solutions = [
    [0, 1, 2],  # Identity mapping
    [2, 0, 1],  # Our previous example
    [1, 2, 0]   # Another permutation
]

print("\nComparing Different Solutions:")
for sol in solutions:
    cost, _ = calculate_fitness_with_details(flow_matrix, distance_matrix, sol)
    print(f"Solution {sol}: Cost = {cost}")