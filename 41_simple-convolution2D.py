import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    if padding > 0:
        padded_input = np.pad(input_matrix, 
                              ((padding, padding), (padding, padding)), 
                              mode='constant', constant_values=0)
    else:
        padded_input = input_matrix

    padded_height, padded_width = padded_input.shape

    output_height = ((padded_height - kernel_height) // stride) + 1
    output_width = ((padded_width - kernel_width) // stride) + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(0, padded_height - kernel_height + 1, stride):
        for j in range(0, padded_width - kernel_width + 1, stride):
            
            region = padded_input[i:i+kernel_height, j:j+kernel_width]
            
            val = np.sum(region * kernel)
            
            out_i = i // stride
            out_j = j // stride
            output_matrix[out_i, out_j] = val

    return output_matrix
