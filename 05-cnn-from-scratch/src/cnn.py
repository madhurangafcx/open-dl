from numpy import strings
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


# =====================================================================
# 1. Baseline: Pure Unpadded Convolution (padding = 0 only)
# =====================================================================
def conv2d_unpadded(image: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int | tuple | list | np.ndarray = 1) -> np.ndarray:
    """
    Computes valid 2D discrete convolution (cross-correlation) with no padding (padding = 0).
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Parse stride into separate vertical and horizontal step sizes
    if isinstance(stride, (int, float)):
        stride_height = stride_width = int(stride)
    else:
        stride_array = np.asarray(stride).flatten()
        if stride_array.size == 1:
            stride_height = stride_width = int(stride_array[0])
        elif stride_array.size == 2:
            stride_height, stride_width = int(stride_array[0]), int(stride_array[1])
        else:
            raise ValueError(f"stride must contain 1 or 2 values, got {stride_array}")

    # 3. Calculate output dimensions: (Input_dim - Kernel_dim) // Stride + 1
    # For a 5x5 image, 3x3 kernel, and stride (2, 2):
    # out_h = (5 - 3) // 2 + 1 = 2 rows
    # out_w = (5 - 3) // 2 + 1 = 2 columns
    output_height = (image_height - kernel_height) // stride_height + 1
    output_width = (image_width - kernel_width) // stride_width + 1

    # 4. Pre-allocate the output feature map with zeros
    feature_map = np.zeros((output_height, output_width), dtype=np.float32)

    # THIS CAN BE WRITE IN BOTH WAYS ------
    # First Way
    # feature_map = []
    # for _ in range(output_height):
    #     row = []
    #     for _ in range(output_width):
    #         row.append(0.0)
    #     feature_map.append(row)
    # feature_map = np.array(feature_map, dtype=np.float32)

    # Second Way
    # feature_map = [[0.0] * output_width for _ in range(output_height)]
    # feature_map = np.array(feature_map, dtype=np.float32)

    # 5. Slide the window across the image using s_h (row step) and s_w (col step)
    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride_height
            col_start = j * stride_width
            patch = image[row_start:row_start + kernel_height, col_start:col_start + kernel_width]
            feature_map[i, j] = np.sum(patch * kernel) + bias

    return feature_map


# =====================================================================
# 2. General: Convolution with Flexible Padding (padding = 0 or 1)
# =====================================================================
def conv2d(image: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int | tuple | list | np.ndarray = 1, padding: int | tuple | list | np.ndarray = 0) -> np.ndarray:
    """
    Computes 2D discrete convolution (cross-correlation) supporting arbitrary stride and zero-padding.
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Parse stride
    if isinstance(stride, (int, float)):
        stride_height = stride_width = int(stride)
    else:
        stride_array = np.asarray(stride).flatten()
        if stride_array.size == 1:
            stride_height = stride_width = int(stride_array[0])
        elif stride_array.size == 2:
            stride_height, stride_width = int(stride_array[0]), int(stride_array[1])
        else:
            raise ValueError(f"stride must contain 1 or 2 values, got {stride_array}")

    # Parse padding
    if isinstance(padding, (int, float)):
        padding_height = padding_width = int(padding)
    else:
        padding_array = np.asarray(padding).flatten()
        if padding_array.size == 1:
            padding_height = padding_width = int(padding_array[0])
        elif padding_array.size == 2:
            padding_height, padding_width = int(padding_array[0]), int(padding_array[1])
        else:
            raise ValueError(f"padding must contain 1 or 2 values, got {padding_array}")

    # Validate padding is not negative
    if padding_height < 0 or padding_width < 0:
        raise ValueError(f"padding cannot be negative, got ({padding_height}, {padding_width})")

    # Create the padded input image
    if padding_height > 0 or padding_width > 0:
        padded_height = image_height + 2 * padding_height
        padded_width = image_width + 2 * padding_width
        padded_image = np.zeros((padded_height, padded_width), dtype=np.float32)
        padded_image[padding_height:padding_height + image_height, padding_width:padding_width + image_width] = image
    else:
        padded_image = image

    # Calculate output dimensions using the padded image dimensions
    padded_h, padded_w = padded_image.shape
    output_height = (padded_h - kernel_height) // stride_height + 1
    output_width = (padded_w - kernel_width) // stride_width + 1

    # Pre-allocate output feature map
    feature_map = np.zeros((output_height, output_width), dtype=np.float32)

    # Slide the window across the padded_image
    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride_height
            col_start = j * stride_width
            patch = padded_image[row_start:row_start + kernel_height, col_start:col_start + kernel_width]
            feature_map[i, j] = np.sum(patch * kernel) + bias

    return feature_map

def max_pool2d(image: np.ndarray, pool_size: int | tuple = 2, stride: int | tuple | None = None) -> np.ndarray:
    image_height, image_width = image.shape

    # If pool_size is integer set both pool_height and pool_width to that value
    pool_height = pool_width = int(pool_size) if isinstance(pool_size, (int, float)) else pool_size

    # default stride equals pool_size (non-overlapping windows)
    if stride is None:
        stride_height, stride_width = pool_height, pool_width
    else:
        stride_height = stride_width = int(stride) if isinstance(stride, (int, float)) else stride

    output_height = (image_height - pool_height) // stride_height + 1
    output_width = (image_width - pool_width) // stride_width + 1

    pooled_map = np.zeros((output_height, output_width), dtype=np.float32)

    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride_height
            col_start = j * stride_width
                
            pool_window = image[row_start:row_start + pool_height, col_start:col_start + pool_width]
            pooled_map[i, j] = np.max(pool_window)

    return pooled_map

def avg_pool2d(image: np.ndarray, pool_size: int | tuple = 2, stride: int | tuple | None = None) -> np.ndarray:
    image_height, image_width = image.shape
    pool_height = pool_width = int(pool_size) if isinstance(pool_size, (int, float)) else pool_size

    if stride is None:
        stride_height, stride_width = pool_height, pool_width
    else:
        stride_height = stride_width = int(stride) if isinstance(stride, (int, float)) else stride

    output_height = (image_height - pool_height) // stride_height + 1
    output_width = (image_width - pool_width) // stride_width + 1

    pooled_map = np.zeros((output_height, output_width), dtype=np.float32)

    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride_height
            col_start = j * stride_width
            pool_window = image[row_start : row_start + pool_height, col_start : col_start + pool_width]
            
            # --- ONLY CHANGE: Use np.mean instead of np.max ---
            pooled_map[i, j] = np.mean(pool_window)

    return pooled_map


def min_pool2d(image: np.ndarray, pool_size: int | tuple = 2, stride: int | tuple | None = None) -> np.ndarray:
    image_height, image_width = image.shape
    pool_height = pool_width = int(pool_size) if isinstance(pool_size, (int, float)) else pool_size

    if stride is None:
        stride_height, stride_width = pool_height, pool_width
    else:
        stride_height = stride_width = int(stride) if isinstance(stride, (int, float)) else stride

    output_height = (image_height - pool_height) // stride_height + 1
    output_width = (image_width - pool_width) // stride_width + 1

    pooled_map = np.zeros((output_height, output_width), dtype=np.float32)

    for i in range(output_height):
        for j in range(output_width):
            row_start = i * stride_height
            col_start = j * stride_width
            pool_window = image[row_start : row_start + pool_height, col_start : col_start + pool_width]
            
            # --- ONLY CHANGE: Use np.min instead of np.max ---
            pooled_map[i, j] = np.min(pool_window)

    return pooled_map


if __name__ == "__main__":
    # Baseline call: Pure Unpadded Convolution
    output = conv2d_unpadded(image, kernel, bias)

    print("\n=== Input Image (5x5) ===")
    print(image)

    print("\n=== Kernel / Filter (3x3) ===")
    print(kernel)

    print(f"\n=== Convolved Feature Map ({output.shape[0]}x{output.shape[1]}) ===")
    print(output)

    # -----------------------------------------------------------------
    # Call 1: Valid Padding (padding = 0) -> Output shrinks to 3x3
    # -----------------------------------------------------------------
    output_pad0 = conv2d(image, kernel, bias, stride=1, padding=0)
    print(f"\n=== Output with padding=0 ({output_pad0.shape[0]}x{output_pad0.shape[1]}) ===")
    print(output_pad0)

    # Raw mathematical breakdown for position [0, 0] with padding=0
    print("\n--- Raw Math Breakdown for Position [0, 0] (padding=0) ---")
    first_patch = image[0:3, 0:3]
    element_products = first_patch * kernel

    print("\n1. Extracted 3x3 Patch:")
    print(first_patch)

    print("\n2. Element-wise Multiplication (patch * kernel):")
    print(element_products)

    print(f"\n3. Sum of all 9 elements: {np.sum(element_products)}")
    print(f"\n4. Add bias ({bias}): {np.sum(element_products) + bias} -> Written to feature_map[0, 0]")

    # -----------------------------------------------------------------
    # Comparison Summary
    # -----------------------------------------------------------------
    print("\n--- Value at [0, 0] Comparison ---")
    print(f"padding=0 at [0, 0]: {output_pad0[0, 0]}")

    # -----------------------------------------------------------------
    # Call 2: Same Padding (padding = 1) -> Output stays 5x5
    # -----------------------------------------------------------------
    output_pad1 = conv2d(image, kernel, bias, stride=1, padding=1)
    print(f"\n=== Output with padding=1 ({output_pad1.shape[0]}x{output_pad1.shape[1]}) ===")
    print(output_pad1)

    # Raw mathematical breakdown for position [0, 0] with padding=1
    print("\n--- Raw Math Breakdown for Position [0, 0] (padding=1) ---")
    padded_canvas = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.float32)
    padded_canvas[1:6, 1:6] = image
    print("\n1. Extracted 7x7 Padded Canvas (Original 5x5 with 1-pixel zero border):")
    print(padded_canvas)

    first_patch_pad1 = padded_canvas[0:3, 0:3]
    element_products_pad1 = first_patch_pad1 * kernel
    print("\n2. Extracted 3x3 Patch at [0, 0] from padded canvas (touches zero border):")
    print(first_patch_pad1)

    print("\n3. Element-wise Multiplication (zero padding evaluates to 0 * weight = 0):")
    print(element_products_pad1)

    print(f"\n4. Sum of all 9 elements: {np.sum(element_products_pad1)}")
    print(f"\n5. Add bias ({bias}): {np.sum(element_products_pad1) + bias} -> Written to feature_map[0, 0]")

    # -----------------------------------------------------------------
    # Comparison Summary
    # -----------------------------------------------------------------
    print("\n--- Value at [0, 0] Comparison ---")
    print(f"padding=1 at [0, 0]: {output_pad1[0, 0]}\n")

    # -----------------------------------------------------------------
    # Pooling Comparison: Max vs. Avg vs. Min (pool_size=2, stride=2)
    # -----------------------------------------------------------------
    max_pooled = max_pool2d(output_pad1, pool_size=2, stride=2)
    avg_pooled = avg_pool2d(output_pad1, pool_size=2, stride=2)
    min_pooled = min_pool2d(output_pad1, pool_size=2, stride=2)
    print("\n=== Max Pooled Map (2x2) ===")
    print(max_pooled)
    print("\n=== Average Pooled Map (2x2) ===")
    print(avg_pooled)
    print("\n=== Min Pooled Map (2x2) ===")
    print(min_pooled)
    

