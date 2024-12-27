import numpy as np

# Example arrays
N, M = 5, 5  # Dimensions
information = np.random.randint(0, 100, (N, M))  # Random integers as example data
visibility = np.random.randint(0, 2, (N, M))  # Random binary mask (0 or 1)

# Initialize filtered array with -1
filtered_info = np.full((N, M), -1)

# Update positions where visibility is 0
filtered_info[visibility == 0] = information[visibility == 0]

# Display results
print("Information Array:")
print(information)
print("\nVisibility Array:")
print(visibility)
print("\nFiltered Information (same shape, initialized with -1):")
print(filtered_info)
