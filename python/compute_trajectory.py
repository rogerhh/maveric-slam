import numpy as np
import argparse
import os
from copy import deepcopy

def write_ply(filename, points):
    """Write a PLY file with vertices, edges, and colors."""
    print(points)
    num_points = len(points)
    edges = []
    for i in range(num_points - 1):
        edges.append([i, i+1])
    colors = [[255, 0, 0]]
    for i in range(1, num_points - 1):
        colors.append([0, 0, 255])

    colors.append([0, 0, 0])

    num_edges = len(edges)
    
    with open(filename, 'w') as file:
        # Write the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {num_points}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element edge {num_edges}\n")
        file.write("property int vertex1\n")
        file.write("property int vertex2\n")
        file.write("end_header\n")
        
        # Write the points with colors
        for point, color in zip(points, colors):
            file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        
        # Write the edges
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]}\n")

def load_transform(filename):
    """Load a 4x4 transformation matrix from a .npy file."""
    return np.load(filename)

def save_pose(filename, pose):
    """Save the 3x4 part of a 4x4 matrix to a text file."""
    np.savetxt(filename, pose[:3, :], fmt='%.6f')

def main(start_frame, end_frame, pose_dir, out_dir):
    # Initialize the first pose as the identity matrix
    current_pose = np.eye(4)

    pose_filename = f"{out_dir}/frame-{start_frame:06d}.pose.txt"
    save_pose(pose_filename, current_pose)

    points = []
    points.append(deepcopy(np.squeeze(current_pose[:3, 3])))
    
    for i in range(start_frame, end_frame):
        next_frame = i + 1
        transform_filename = f"{pose_dir}/transform_{i:06d}_{next_frame:06d}.npy"
        
        if os.path.exists(transform_filename):
            # Load the transformation matrix
            transform = np.eye(4)
            transform[:3, :] = load_transform(transform_filename)

            print("transform = ", transform[:3, :3])
            print("cur = ", current_pose[:3, :3])
            
            # Update the current pose
            current_pose[:3, :3] = transform[:3, :3] @ current_pose[:3, :3] 
            current_pose[:3, 3] = transform[:3, 3] + current_pose[:3, 3] 

            # current_pose = np.dot(current_pose, transform)
            print(current_pose[:3, 3])
            points.append(deepcopy(np.squeeze(current_pose[:3, 3])))
            
            # Save the current pose to a file
            pose_filename = f"{out_dir}/frame-{next_frame:06d}.pose.txt"
            save_pose(pose_filename, current_pose)
        else:
            print(f"File {transform_filename} not found. Skipping.")

    print("Write trajectory")
    write_ply(f"{out_dir}/trajectory_{start_frame:06d}_{end_frame:06d}.ply", points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a trajectory from transformation matrices.")
    parser.add_argument("start_frame", type=int, help="The starting frame number.")
    parser.add_argument("end_frame", type=int, help="The ending frame number.")
    parser.add_argument("--pose_dir", type=str, help="Directory that contains all the relative transforms")
    parser.add_argument("--out_dir", type=str, help="Directory of all the outputs")
    
    args = parser.parse_args()
    
    main(args.start_frame, args.end_frame, args.pose_dir, args.out_dir)

