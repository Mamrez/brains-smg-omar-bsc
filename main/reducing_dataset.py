import numpy as np
import matplotlib.pyplot as plt

def reduce_datapoints(file_path: str, start_index: int, end_index: int, activation_electrode_no: int, readout_electrode_no: int) -> None:
    
    data = np.loadtxt(file_path)
    
    # Ensure valid indices
    if start_index < 0 or end_index > len(data) or start_index >= end_index:
        raise ValueError("Invalid range of data points")

    # Extract the data points from the specified range
    reduced_data = data[start_index:end_index, :]

    # Save the reduced data to a new file
    num_points = end_index - start_index
    new_file_path = f"IO_{num_points}.dat"
    header = f"# Input 0, Input 1, Input 2, Input 3, Input 4, Input 5, Input 6, Output 0\n"
    
    with open(new_file_path, 'w') as f:
        f.write(header)
        np.savetxt(f, reduced_data)
    
    print(f"Reduced data saved to {new_file_path}")

    # Plot the reduced data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot inputs
    for i in range(7):  # Assuming the first 7 columns are inputs
        ax1.plot(reduced_data[:, i], label=f'Input {i}')
    ax1.set_ylabel('Input Values')
    ax1.set_title('Input Data')
    ax1.legend()
    ax1.grid(True)
    
    # Plot output
    ax2.plot(reduced_data[:, 7], label='Output 0', color='r')  # Assuming the 8th column is the output
    ax2.set_xlabel('Data Points')
    ax2.set_ylabel('Output Value')
    ax2.set_title('Output Data')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


reduce_datapoints("main\mainSamplingData\IO.dat", 990000, 1000000, 7, 1)
