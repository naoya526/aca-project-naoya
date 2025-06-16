#mpiexec -n 2 python3 wrapper_test.py
#mpiexec -n 4 python3 wrapper_test.py
import ctypes
import numpy as np
from mpi4py import MPI
import time

# パディングとストライドの設定
P = 1  # 例: パディング1
S = 1  # 例: ストライド1

# ライブラリ読み込み
lib = ctypes.CDLL('./libconv.so')
lib.conv2d_forward.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 8
lib.conv2d_forward.restype = None

# MPIの初期化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 全体のデータサイズ
B, IC, OC, H, W, K = 10000, 1, 1, 32, 32, 3

# 出力サイズ計算
out_H = (H + 2 * P - K) // S + 1
out_W = (W + 2 * P - K) // S + 1

# 各プロセスの処理範囲（batch axisで分割）
B_local = B // size
start_idx = rank * B_local
end_idx = start_idx + B_local

# ランク0のみが全データを持つ（入力をBroadcastする場合は共有でもOK）
if rank == 0:
    input_np_full = np.random.rand(B, IC, H, W).astype(np.float32)
    kernel_np = np.random.rand(OC, IC, K, K).astype(np.float32)
else:
    input_np_full = None
    kernel_np = np.empty((OC, IC, K, K), dtype=np.float32)

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
output_local = np.zeros((B_local, OC, out_H, out_W), dtype=np.float32)

# C関数呼び出し
start = time.perf_counter()
lib.conv2d_forward(
    input_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    output_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B_local, IC, OC, H, W, K, P, S
)
end = time.perf_counter()
elapsed = end - start

# 結果をランク0にGatherして統合
if rank == 0:
    output_np_full = np.empty((B, OC, out_H, out_W), dtype=np.float32)
else:
    output_np_full = None

comm.Gather(output_local, output_np_full, root=0)

# max時間をrootで表示
max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)

if rank == 0:
    print("Output shape:", output_np_full.shape)
    print(f"Executed time (max across ranks):) {max_time:.6f} sec")
    print(f"Executed time (max across ranks): {max_time:.6f} sec")