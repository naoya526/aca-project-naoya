#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

extern void conv2d_forward(float* input, float* kernel, float* output,
                           int B, int IC, int OC, int H, int W, int K);

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int B = 160000, IC = 1, OC = 1, H = 32, W = 32, K = 3;
    const int B_local = B / size;
    const int out_H = H - K + 1, out_W = W - K + 1;

    float *input_full = NULL, *output_full = NULL;
    float *input_local = (float*)malloc(B_local * IC * H * W * sizeof(float));
    float *output_local = (float*)malloc(B_local * OC * out_H * out_W * sizeof(float));
    float *kernel = (float*)malloc(OC * IC * K * K * sizeof(float));
    // ランク0でデータ作成
    if (rank == 0) {
        input_full = (float*)malloc(B * IC * H * W * sizeof(float));
        output_full = (float*)malloc(B * OC * out_H * out_W * sizeof(float));
        for (int i = 0; i < B * IC * H * W; ++i) input_full[i] = rand_float();
        for (int i = 0; i < OC * IC * K * K; ++i) kernel[i] = rand_float();
    }

    time_t start_time, end_time;
    if (rank ==0){
        start_time = time(NULL);
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // 全プロセス同期
    start_time = time(NULL);
    double start = MPI_Wtime();
    // カーネルのBroadcast
    MPI_Bcast(kernel, OC * IC * K * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 入力データの分散
    MPI_Scatter(input_full, B_local * IC * H * W, MPI_FLOAT,
                input_local, B_local * IC * H * W, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // 畳み込み処理と時間計測
    conv2d_forward(input_local, kernel, output_local, B_local, IC, OC, H, W, K);

    // 結果を集める
    MPI_Gather(output_local, B_local * OC * out_H * out_W, MPI_FLOAT,
               output_full, B_local * OC * out_H * out_W, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    

    // 最大時間をrank 0で集約表示
    double end = MPI_Wtime();
    double elapsed = end - start;

    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Output shape: (%d, %d, %d, %d)\n", B, OC, out_H, out_W);
        printf("Executed time (max across ranks): %.6f sec\n", max_time);
        end_time = time(NULL);
        double act_time = end_time - start_time;
        printf("集約完了までの時間: %.6f sec\n",act_time );
    }

    // 後始末
    if (rank == 0) { free(input_full); free(output_full); }
    free(input_local); free(output_local); free(kernel);

    MPI_Finalize();
    return 0;
}
