#mpicc -fPIC -shared -o libconv.so conv.c
#mpiexec -n 4 python3 wrapper_test_imageprocess.py
import ctypes
import numpy as np
from mpi4py import MPI
import time
from PIL import Image
import matplotlib.pyplot as plt
import psutil
import os
import sys
# ---- 設定 ----
IMG_DIR = "pic/"
IC, OC, H, W, K, P, S = 3, 3, 1024, 1024, 3, 1, 1
SEED = 200

def parse_batch_size(default=40):
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            print("Invalid batch size. Using default B=40.")
    return default

def load_input_images(batch_size, ic, h, w, img_dir):
    input_np = np.empty((batch_size, ic, h, w), dtype=np.float32)
    for i in range(batch_size):
        img_path = os.path.join(img_dir, f"input_{i}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32)
            img_np = np.transpose(img_np, (2, 0, 1))
        except FileNotFoundError:
            print(f"Image {img_path} not found. Using random data instead.")
            img_np = np.random.rand(ic, h, w).astype(np.float32)
        input_np[i] = img_np
    return input_np

def create_kernel(oc, ic, k, identity=False):
    if identity:
        kernel = np.zeros((oc, ic, k, k), dtype=np.float32)
        center = k // 2
        for c in range(min(oc, ic)):
            kernel[c, c, center, center] = 1.0
        return kernel
    else:
        return np.random.rand(oc, ic, k, k).astype(np.float32)

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
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)

def main():
    # ---- MPI初期化 ----
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    np.random.seed(SEED)

    # ---- バッチサイズ取得 ----
    B = parse_batch_size()
    B_local = B // size

    # ---- Cライブラリの読み込み ----
    lib = ctypes.CDLL('./libconv.so')
    lib.conv2d_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 8
    lib.conv2d_forward.restype = None

    # ---- データ準備 ----
    if rank == 0:
        input_np_full = load_input_images(B, IC, H, W, IMG_DIR)
        kernel_np = create_kernel(OC, IC, K, identity=False)
    else:
        input_np_full = None
        kernel_np = np.empty((OC, IC, K, K), dtype=np.float32)

    # ---- メモリ使用量収集 ----
    mem_usage = get_memory_usage_mb()
    all_mem_usage = comm.gather(mem_usage, root=0)

    # ---- カーネルのブロードキャスト ----
    comm.Bcast(kernel_np, root=0)

    # ---- 入力データの分散 ----
    input_local = np.empty((B_local, IC, H, W), dtype=np.float32)
    comm.Scatter([input_np_full, MPI.FLOAT], input_local, root=0)

    # ---- 出力配列の確保 ----
    out_H = (H + 2 * P - K) // S + 1
    out_W = (W + 2 * P - K) // S + 1
    output_local = np.zeros((B_local, OC, out_H, out_W), dtype=np.float32)

    # ---- CのConv2d実行 ----
    start = time.perf_counter()
    lib.conv2d_forward(
        input_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_local, IC, OC, H, W, K, P, S
    )
    elapsed = time.perf_counter() - start

    # ---- 出力の集約 ----
    output_np_full = np.empty((B, OC, out_H, out_W), dtype=np.float32) if rank == 0 else None
    comm.Gather(output_local, output_np_full, root=0)

    # ---- 最大実行時間の集約 ----
    max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)

    # ---- Rank0で出力 ----
    if rank == 0:
        print("Output shape:", output_np_full.shape)
        print(f"Executed time (max across ranks): {max_time:.6f} sec")
        print("Memory usage in each rank(MB):", all_mem_usage)
        print(f"Sum of memory usage: {sum(all_mem_usage):.2f} MB")
        # 最初の出力画像を保存・表示
        output_img = normalize_output_for_display(output_np_full[0])
        display_and_save_output(output_img, "Output (rank 0, batch 0, RGB)", "output_rank0_batch0_rgb.png")

if __name__ == "__main__":
    main()