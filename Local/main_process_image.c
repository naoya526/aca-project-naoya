// mpicc -o cnn_mpi main_process_image.c conv.c -lm
// mpiexec -n 4 ./cnn_mpi

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

extern void conv2d_forward(float* input, float* kernel, float* output,
                           int B, int IC, int OC, int H, int W, int K, int P, int S);

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

void read_or_generate_input(float *input_full, int total_size, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (fp) {
        size_t read = fread(input_full, sizeof(float), total_size, fp);
        if (read != total_size) {
            fprintf(stderr, "Failed to read full image data from %s\n", path);
            fclose(fp);
            exit(1);
        }
        fclose(fp);
    } else {
        fprintf(stderr, "Failed to open %s for reading. Using random data.\n", path);
        for (int i = 0; i < total_size; ++i) input_full[i] = rand_float();
    }
}

void fill_random(float *arr, int size) {
    for (int i = 0; i < size; ++i) arr[i] = rand_float();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // パラメータ
    const int B = 40, IC = 3, OC = 3, H = 1024, W = 1024, K = 3, P = 1, S = 1;
    const int B_local = B / size;
    const int out_H = (H + 2 * P - K) / S + 1;
    const int out_W = (W + 2 * P - K) / S + 1;

    float *input_full = NULL, *output_full = NULL;
    float *input_local = (float*)malloc(B_local * IC * H * W * sizeof(float));
    float *output_local = (float*)malloc(B_local * OC * out_H * out_W * sizeof(float));
    float *kernel = (float*)malloc(OC * IC * K * K * sizeof(float));

    // ランク0でデータ作成
    if (rank == 0) {
        input_full = (float*)malloc(B * IC * H * W * sizeof(float));
        output_full = (float*)malloc(B * OC * out_H * out_W * sizeof(float));
        read_or_generate_input(input_full, B * IC * H * W, "input_image.bin");
        fill_random(kernel, OC * IC * K * K);
    }

    // タイミング計測
    time_t wall_start = 0, wall_end = 0;
    if (rank == 0) wall_start = time(NULL);

    MPI_Barrier(MPI_COMM_WORLD); // 全プロセス同期
    double start = MPI_Wtime();

    // カーネルのBroadcast
    MPI_Bcast(kernel, OC * IC * K * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 入力データの分散
    MPI_Scatter(input_full, B_local * IC * H * W, MPI_FLOAT,
                input_local, B_local * IC * H * W, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // 畳み込み処理
    conv2d_forward(input_local, kernel, output_local, B_local, IC, OC, H, W, K, P, S);

    // 結果を集める
    MPI_Gather(output_local, B_local * OC * out_H * out_W, MPI_FLOAT,
               output_full, B_local * OC * out_H * out_W, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double elapsed = end - start;

    // 最大時間をrank 0で集約表示
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        wall_end = time(NULL);
        printf("Output shape: (%d, %d, %d, %d)\n", B, OC, out_H, out_W);
        printf("Executed time (max across ranks): %.6f sec\n", max_time);
        printf("集約完了までの時間: %.6f sec\n", difftime(wall_end, wall_start));
    }

    // 後始末
    if (rank == 0) { free(input_full); free(output_full); }
    free(input_local); free(output_local); free(kernel);

    MPI_Finalize();
    return 0;
}
