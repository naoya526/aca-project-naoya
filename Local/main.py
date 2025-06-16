#mpicc -fPIC -shared -o libconv.so conv.c
#mpiexec -n 4 python3 main.py
import ctypes
import numpy as np
from mpi4py import MPI
import time
from PIL import Image
from matplotlib import pyplot as plt
import psutil
import os
import sys
# ---- 設定 ----
IMG_DIR = "pic/"
IC, OC, H, W, K =  3, 3, 1024, 1024, 3
SEED = 200



# ---- Cライブラリの読み込み ----
lib = ctypes.CDLL('./libconv.so')
lib.conv2d_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 6
lib.conv2d_forward.restype = None

# ---- MPI初期化 ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if len(sys.argv) > 1:
        try:
            B = int(sys.argv[1])
        except ValueError:
            B = 32  # デフォルト値
    else:
        B = 32  # デフォルト値
else:
    B = None

B = comm.bcast(B, root=0)

B_local = B // size
start_idx = rank * B_local
end_idx = start_idx + B_local

np.random.seed(SEED)

def load_input_images(batch_size):
    input_np = np.empty((batch_size, IC, H, W), dtype=np.float32)
    for i in range(batch_size):
        try:
            img = Image.open(f"{IMG_DIR}input_{i}.jpg").convert("RGB")
            img_np = np.array(img, dtype=np.float32)
            img_np = np.transpose(img_np, (2, 0, 1))
        except FileNotFoundError:
            #print(f"Warning: input_{i}.jpeg not found, generating random image")
            img_np = np.random.rand(IC, H, W).astype(np.float32)
        input_np[i] = img_np
    return input_np

# ---- kernel ----
def create_identity_kernel():
    kernel = np.zeros((OC, IC, K, K), dtype=np.float32)
    center = K // 2
    for c in range(min(OC, IC)):
        kernel[c, c, center, center] = 1.0
    return kernel

def create_random_kernel():
    kernel = np.random.rand(OC, IC, K, K).astype(np.float32)
    return kernel

# ---- Normalize output ----
def normalize_output_for_display(arr):
    arr_norm = np.zeros_like(arr)
    for c in range(arr.shape[0]):
        channel = arr[c]
        arr_norm[c] = 255 * (channel - channel.min()) / (channel.ptp() + 1e-8)
    return arr_norm.astype(np.uint8)

def display_and_save_output(output_arr, title, filename):
    arr_rgb = np.transpose(output_arr, (1, 2, 0))
    Image.fromarray(arr_rgb, mode="RGB").save(filename)
    plt.imshow(arr_rgb)
    plt.title(title)
    plt.colorbar()
    plt.show()

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 ** 2)  # MB
# ---- data preparation ----
if rank == 0:
    input_np_full = load_input_images(B)
    #kernel_np = create_identity_kernel()
    kernel_np = create_random_kernel()
else:
    input_np_full = None
    kernel_np = np.empty((OC, IC, K, K), dtype=np.float32)

# ---- gather memory usage ----
mem_usage = get_memory_usage_mb()
all_mem_usage = comm.gather(mem_usage, root=0)

# ---- broadcast kernel ----
comm.Bcast(kernel_np, root=0)

# ---- scatter input data ----
input_local = np.empty((B_local, IC, H, W), dtype=np.float32)
comm.Scatter([input_np_full, MPI.FLOAT], input_local, root=0)

# ---- allocate output ----
output_local = np.zeros((B_local, OC, H - K + 1, W - K + 1), dtype=np.float32)

# ---- Execute Conv2d in C ----
start = time.perf_counter()
lib.conv2d_forward(
    input_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    output_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B_local, IC, OC, H, W, K
)
elapsed = time.perf_counter() - start

# ---- Gather the output ----
output_np_full = np.empty((B, OC, H - K + 1, W - K + 1), dtype=np.float32) if rank == 0 else None
comm.Gather(output_local, output_np_full, root=0)

# ---- show maximum execution time in Rank 0 ----
max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)

# ---- show output（Rank0） ----
if rank == 0:
    print("Output shape:", output_np_full.shape)
    print(f"Executed time (max across ranks): {max_time:.6f} sec")
    print("Memory usage in each rank(MB):", all_mem_usage)
    print(f"Sum of memory usage: {sum(all_mem_usage):.2f} MB")
    # first output image (rank 0, batch 0)
    output_img = normalize_output_for_display(output_np_full[0])
    display_and_save_output(output_img, "Output (rank 0, batch 0, RGB)", "output_rank0_batch0_rgb.png")