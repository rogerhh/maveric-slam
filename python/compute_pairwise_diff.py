import numpy as np
import argparse

def read_trajectory(filename):
    """Reads a trajectory file where each line has 12 elements."""
    trajectory = []
    with open(filename, 'r') as file:
        for line in file:
            elements = list(map(float, line.strip().split()))
            if len(elements) == 12:
                matrix = np.array(elements).reshape(3, 4)
                trajectory.append(matrix)
    return trajectory

def compute_relative_transform(matrix1, matrix2):
    """Computes the relative transformation between two 3x4 transformation matrices."""
    # Convert 3x4 matrices to 4x4 by adding a bottom row [0, 0, 0, 1]
    matrix1_4x4 = np.vstack([matrix1, [0, 0, 0, 1]])
    matrix2_4x4 = np.vstack([matrix2, [0, 0, 0, 1]])
    
    # Compute the relative transformation
    relative_transform = matrix2_4x4 @ np.linalg.inv(matrix1_4x4) # @ matrix2_4x4
    
    # Extract the rotation and translation from the 4x4 relative transformation matrix
    rotation = relative_transform[:3, :3]
    translation = relative_transform[:3, 3]
    
    return rotation, translation

def print_relative_transforms(trajectory):
    """Prints the relative transform between each line and the next."""
    for i in range(len(trajectory) - 1):
        rotation, translation = compute_relative_transform(trajectory[i], trajectory[i + 1])
        
        # Flatten the rotation matrix and translation vector for printing
        rotation_flat = rotation.flatten()
        translation_flat = translation.flatten()
        
        # Print the rotation and translation on the same line
        print("Rotation:", " ".join(f"{val:.6f}" for val in rotation_flat), 
              "| Translation:", " ".join(f"{val:.6f}" for val in translation_flat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("infile", type=str, help="The input file path.")

    args = parser.parse_args()

    trajectory = read_trajectory(args.infile)
    
    print_relative_transforms(trajectory)

