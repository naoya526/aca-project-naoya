#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

// paddingとstrideを引数に追加
void conv2d_forward(float* input, float* kernel, float* output,
                    int batch_size, int in_channels, int out_channels,
                    int height, int width, int kernel_size,
                    int padding, int stride) {
    // input: [batch_size, in_channels, height, width]
    // kernel: [out_channels, in_channels, kernel_size, kernel_size]
    // output: [batch_size, out_channels, out_height, out_width]

    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width  = (width  + 2 * padding - kernel_size) / stride + 1;

    // MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    int local_batch = batch_size / size;
    int start = rank * local_batch;
    int end = start + local_batch;

    // パディングした入力を作成
    int padded_height = height + 2 * padding;
    int padded_width  = width + 2 * padding;
    float* input_padded = (float*)calloc(local_batch * in_channels * padded_height * padded_width, sizeof(float));

    // パディングを適用
    for (int b = 0; b < local_batch; ++b) {
        for (int c = 0; c < in_channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int src_idx = b * in_channels * height * width
                                + c * height * width
                                + h * width
                                + w;
                    int dst_idx = b * in_channels * padded_height * padded_width
                                + c * padded_height * padded_width
                                + (h + padding) * padded_width
                                + (w + padding);
                    input_padded[dst_idx] = input[src_idx];
                }
            }
        }
    }

    // 畳み込み本体
    for (int b = 0; b < local_batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                int in_idx = b * in_channels * padded_height * padded_width
                                           + ic * padded_height * padded_width
                                           + ih * padded_width
                                           + iw;
                                int k_idx = oc * in_channels * kernel_size * kernel_size
                                          + ic * kernel_size * kernel_size
                                          + kh * kernel_size
                                          + kw;
                                sum += input_padded[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                    int out_idx = b * out_channels * out_height * out_width
                                + oc * out_height * out_width
                                + oh * out_width
                                + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    free(input_padded);

    MPI_Barrier(MPI_COMM_WORLD);
}