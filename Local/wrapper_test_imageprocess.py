#mpiexec -n 2 python3 wrapper_test.py
#mpiexec -n 4 python3 wrapper_test.py
import ctypes
import numpy as np
from mpi4py import MPI
import time
from PIL import Image
from matplotlib import pyplot as plt

# ライブラリ読み込み
lib = ctypes.CDLL('./libconv.so')
lib.conv2d_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 6
lib.conv2d_forward.restype = None

# MPIの初期化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Hello")
# 全体のデータサイズ
B, IC, OC, H, W, K = 1, 3, 3, 225, 225, 3
np.random.seed(102)

# 各プロセスの処理範囲（batch axisで分割）
B_local = B // size
start_idx = rank * B_local
end_idx = start_idx + B_local

# ランク0のみが全データを持つ（入力をBroadcastする場合は共有でもOK）
if rank == 0:
    # バッチ数分の画像ファイルを読み込む
    input_np_full = np.empty((B, IC, H, W), dtype=np.float32)
    for i in range(B):
        img_path = "pic/"
        try:
            img = Image.open(img_path+f"input_{i}.jpeg").convert("RGB")  # 32bit float grayscale
            img_np = np.array(img, dtype=np.float32)  # (H, W, 3)
            img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
        except FileNotFoundError:
            print("404 file not found: randomly produced")
            img_np = np.random.rand(H, W).astype(np.float32)
        input_np_full[i, :, :, :] = img_np
    #kernel_np = np.random.rand(OC, IC, K, K).astype(np.float32)
    kernel_np = np.zeros((OC, IC, K, K), dtype=np.float32)
    center = K // 2
    for oc in range(OC):
        for ic in range(IC):
            kernel_np[oc, ic, center, center] = 1.0
else:
    input_np_full = None
    kernel_np = np.empty((OC, IC, K, K), dtype=np.float32)

#debug
if rank == 0:
    plt.imshow(np.transpose(input_np_full[0], (1, 2, 0)).astype(np.uint8))
    plt.title("Input Image (rank 0, batch 0)")
    plt.show()

# カーネルを全プロセスにbroadcast（全プロセスで同じカーネルを使う）
comm.Bcast(kernel_np, root=0)

# 各プロセス用の入力データを確保
input_local = np.empty((B_local, IC, H, W), dtype=np.float32)

# 入力データを分割してscatter（rankごとにバッチを分割）
if rank == 0:
    comm.Scatter([input_np_full, MPI.FLOAT], input_local, root=0)
else:
    comm.Scatter(None, input_local, root=0)

# 出力配列（各プロセス分）
output_local = np.zeros((B_local, OC, H - K + 1, W - K + 1), dtype=np.float32)

# C関数呼び出し
start = time.perf_counter()
lib.conv2d_forward(
    input_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    output_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B_local, IC, OC, H, W, K
)
end = time.perf_counter()
elapsed = end - start

# 結果をランク0にGatherして統合
if rank == 0:
    output_np_full = np.empty((B, OC, H - K + 1, W - K + 1), dtype=np.float32)
    # 画像表示用に最初の出力画像を保存・表示
    # 最初の出力画像を保存
    # 各チャンネルを正規化してRGB画像として保存
    arr = output_local[0]  # shape: (OC, H_out, W_out)
    # 3チャンネル(RGB)に対応している前提
    arr_norm = np.zeros_like(arr)
    for c in range(3):
        channel = arr[c]
        arr_norm[c] = 255 * (channel - channel.min()) / (channel.ptp() + 1e-8)
    arr_norm = arr_norm.astype(np.uint8)
    arr_rgb = np.transpose(arr_norm, (1, 2, 0))  # (H, W, 3)
    out_img = Image.fromarray(arr_rgb, mode="RGB")
    out_img.save("output_rank0_batch0_rgb.png")
    # 例として最初のバッチの出力画像を表示
    plt.imshow(arr_rgb)
    plt.title("Output (rank 0, batch 0, RGB)")
    plt.colorbar()
    plt.show()
else:
    output_np_full = None

comm.Gather(output_local, output_np_full, root=0)

# max時間をrootで表示
max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)

if rank == 0:
    print("Output shape:", output_np_full.shape)
    print(f"Executed time (max across ranks): {max_time:.6f} sec")

