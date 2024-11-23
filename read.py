import numpy as np
from typing import Tuple

def read_tai12a(file_path : str) -> Tuple[np.array]:
    """
    Read tai12a.dat file and return flow and distance matrices
    
    Args:
        file_path: Path to the tai12a.dat file
        
    Returns:
        flow_matrix: numpy array of flow values
        distance_matrix: numpy array of distance values
    """
    with open(file_path, 'r') as file:
        # Read line 1
        n = int(file.readline().strip())
        
        # Initialize zeros matrices
        flow_matrix = np.zeros((n, n))
        distance_matrix = np.zeros((n, n))
        
        # Skip the blank line
        file.readline()
        
        # Read distance matrix
        for i in range(n):
            # Read line and split into numbers
            row = list(map(int, file.readline().strip().split()))
            distance_matrix[i] = row
            
        # Skip the blank line
        file.readline()
        
        # Read flow matrix
        for i in range(n):
            # Read line and split into numbers
            row = list(map(int, file.readline().strip().split()))
            flow_matrix[i] = row
        
    return flow_matrix, distance_matrix