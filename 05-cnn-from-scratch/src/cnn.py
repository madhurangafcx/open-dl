import numpy as np

# Input Image: 5 x 5 (Single channel grayscale)
image = np.array([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1],
], dtype=np.float32)

# Kernel / Filter: 3 x 3
kernel = np.array([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2],
    # dtype=np.float32 explicitly forces NumPy to store every number in the array as a 32-bit floating-point number
    # Hardware Optimization: GPUs are heavily optimized for 32-bit floating-point math.
    # Memory Efficiency: It consumes exactly half the RAM and VRAM compared to Python's standard 64-bit floats
    # Neural Network Standard: 32-bit precision is the strict industry standard for training computer vision architectures
    # Why not 64-bit (double)?
    # 1. Wasted Memory: 64-bit doubles take twice the memory.
    # 2. Slower Training: GPUs perform 32-bit operations much faster.
    # 3. Sufficient Precision: 32-bit is accurate enough for deep learning tasks.
], dtype=np.float32)

# Bias term
bias = 0.0

# image: np.ndarray - Input image matrix.
# kernel: np.ndarray - The convolutional filter (kernel) matrix.
# bias: float - A scalar value added to the result of each convolution.
# stride: int | tuple | list | np.ndarray - Step size of the kernel across the image.
#         Can be an int (e.g. 1) or a 2D pair (e.g. (2, 2) or np.array([2, 2])).
def conv2d(image: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int | tuple | list | np.ndarray = 1) -> np.ndarray:
    """
    Computes valid 2D discrete convolution (cross-correlation) between an image and a filter.

    Mental Model:
    -------------
    Imagine placing a 3x3 cardboard stencil (kernel) on a 5x5 sheet of paper (image).
    At every position:
      1. Look at the 9 numbers visible under the stencil ("patch").
      2. Multiply each patch number by the corresponding stencil number ("patch * kernel").
      3. Sum all 9 products together into a single number + bias ("np.sum(...) + bias").
      4. Write that single number into the output feature map.
      5. Slide the stencil to the right by s_w steps, then down by s_h steps.
    """

    # 1. Get the spatial dimensions of the image (e.g. 5x5) and kernel (e.g. 3x3)
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # 2. Parse stride into separate vertical (s_h) and horizontal (s_w) step sizes.
    # Accepts int, float, tuple, list, or np.ndarray (e.g., 2, (2, 2), np.array([2, 2]))
    if isinstance(stride, (int, float)):
        s_h = s_w = int(stride)
    else:
        # np.asarray handles tuples, lists, and np.ndarrays seamlessly
        stride_arr = np.asarray(stride).flatten()
        if stride_arr.size == 1:
            s_h = s_w = int(stride_arr[0])
        elif stride_arr.size == 2:
            s_h, s_w = int(stride_arr[0]), int(stride_arr[1])
        else:
            raise ValueError(f"stride must contain 1 or 2 values, got {stride_arr}")

    # 3. Calculate output dimensions: (Input_dim - Kernel_dim) // Stride + 1
    # For a 5x5 image, 3x3 kernel, and stride (2, 2):
    # out_h = (5 - 3) // 2 + 1 = 2 rows
    # out_w = (5 - 3) // 2 + 1 = 2 columns
    out_h = (img_h - k_h) // s_h + 1
    out_w = (img_w - k_w) // s_w + 1

    # 4. Pre-allocate the output feature map with zeros
    feature_map = np.zeros((out_h, out_w), dtype=np.float32)

    # 5. Slide the window across the image using s_h (row step) and s_w (col step)
    for i in range(out_h):
        for j in range(out_w):

            # Top-left starting indices for current patch
            r_start = i * s_h
            c_start = j * s_w

            # Extract receptive field patch of size (k_h, k_w)
            patch = image[r_start:r_start + k_h, c_start:c_start + k_w]

            # Element-wise product + sum all elements + bias
            feature_map[i, j] = np.sum(patch * kernel) + bias

    return feature_map


if __name__ == "__main__":
    output = conv2d(image, kernel, bias)

    print("\n=== Input Image (5x5) ===")
    print(image)

    print("\n=== Kernel / Filter (3x3) ===")
    print(kernel)

    print(f"\n=== Convolved Feature Map ({output.shape[0]}x{output.shape[1]}) ===")
    print(output)

    # Detailed breakdown of position (0, 0)
    print("\n--- Math Breakdown for Position [0, 0] (Top-Left) ---")
    first_patch = image[0:3, 0:3]
    element_products = first_patch * kernel

    print("\n1. Extracted 3x3 Patch:")
    print(first_patch)

    print("\n2. Element-wise Multiplication (patch * kernel):")
    print(element_products)

    print(f"\n3. Sum of all 9 elements: {np.sum(element_products)}")
    print(f"\n4. Add bias ({bias}): {np.sum(element_products) + bias} -> Written to feature_map[0, 0]\n")
