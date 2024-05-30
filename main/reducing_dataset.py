import numpy as np

def reduce_datapoints(file_path: str, num_points: int, activation_electrode_no: int, readout_electrode_no: int) -> None:
    # Read the data from the file
    data = np.loadtxt(file_path)
    
    # Ensure we don't ask for more points than available
    if num_points > len(data):
        raise ValueError("Requested number of data points exceeds available data points")

    # Reduce the data points
    reduced_data = data[:num_points, :]

    # Save the reduced data to a new file
    new_file_path = f"IO_{num_points}.dat"
    header = f"# Input 0, Input 1, Input 2, Input 3, Input 4, Input 5, Input 6, Output 0\n"
    
    with open(new_file_path, 'w') as f:
        f.write(header)
        np.savetxt(f, reduced_data)
    
    print(f"Reduced data saved to {new_file_path}")

# Example usage
reduce_datapoints("main\mainSamplingDataFull\IO.dat", 1000000, 7, 1)
