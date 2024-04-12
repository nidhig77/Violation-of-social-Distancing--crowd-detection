import numpy as np


import numpy as np


def main_part(positions, socialdistance):
    social_distancing_matrix = get_respect_social_distancing(positions, socialdistance)
    violators_count = count_violators(social_distancing_matrix)

def get_respect_social_distancing(positions, socialdistance):
    # Calculate the Euclidean distance between each pair of persons
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    
    # Check if the distance between each pair is less than the social_distance
    respecting_social_distancing = distances < socialdistance
    
    return respecting_social_distancing.astype(int)







def count_violators(social_distancing_matrix):
    # Sum along rows and columns to check for violations
    row_violations = np.sum(social_distancing_matrix == 0, axis=1)
    column_violations = np.sum(social_distancing_matrix == 0, axis=0)
    
    # Combine row and column violations and count unique individuals
    total_violations = np.sum(np.logical_or(row_violations > 0, column_violations > 0))
    
    return total_violations


