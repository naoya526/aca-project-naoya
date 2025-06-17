//mpicc -o cnn_mpi main.c conv.c -lm
//mpiexec -n 4 ./cnn_mpi
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

extern void conv2d_forward(float* input, float* kernel, float* output,
                           int B, int IC, int OC, int H, int W, int K);

float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

size_t get_memory_usage_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    char line[256];
    size_t mem = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &mem);
            break;
        }
    }
    fclose(file);
    return mem; // KB
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int B = 10, IC = 1, OC = 1, H = 1024, W = 1024, K = 3;
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
        const char *path = "input_image.bin";
        FILE *fp = fopen(path, "rb");
        if (fp) {
            size_t read = fread(input_full, sizeof(float), B * IC * H * W, fp);
            if (read != B * IC * H * W) {
                fprintf(stderr, "Failed to read full image data from %s\n", path);
                exit(1);
            }
            fclose(fp);
        } else {
            fprintf(stderr, "Failed to open %s for reading. Using random data.\n", path);
            for (int i = 0; i < B * IC * H * W; ++i) input_full[i] = rand_float();
        }
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
    
    // メモリ使用量の取得と集約
    size_t mem_kb = get_memory_usage_kb();
    size_t* all_mem_kb = NULL;
    if (rank == 0) {
        all_mem_kb = (size_t*)malloc(size * sizeof(size_t));
    }
    MPI_Gather(&mem_kb, 1, MPI_UNSIGNED_LONG, all_mem_kb, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Memory usage (VmRSS) per rank (KB):\n");
        for (int i = 0; i < size; ++i) {
            printf("  Rank %d: %zu MB\n", i, all_mem_kb[i]/1024);
        }
        free(all_mem_kb);
    }

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
