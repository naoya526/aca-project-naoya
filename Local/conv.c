#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void conv2d_forward(float* input, float* kernel, float* output,
                    int batch_size, int in_channels, int out_channels,
                    int height, int width, int kernel_size) {
    // input: [batch_size, in_channels, height, width]
    // kernel: [out_channels, in_channels, kernel_size]
    // output: [batch_size, out_channels, height-kernel_size+1, width-kernel_size+1]

    // MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    int local_batch = batch_size / size;
    int start = rank * local_batch;
    int end = start + local_batch;

    printf("[MPI] rank=%d, size=%d, local_batch=%d, start=%d, end=%d\n", rank, size, local_batch, start, end);

    for (int b = start; b < end; ++b) {// for each batch
        for (int oc = 0; oc < out_channels; ++oc) {// for each output channel
            for (int h = 0; h < height - kernel_size + 1; ++h) {// for each output height
                for (int w = 0; w < width - kernel_size + 1; ++w) {// for each output width
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {// for each input channel
                        for (int kh = 0; kh < kernel_size; ++kh) {// for each kernel height
                            for (int kw = 0; kw < kernel_size; ++kw) {// for each kernel width
                                int ih = h + kh;
                                int iw = w + kw;
                                sum += input[b*in_channels*height*width + ic*height*width + ih*width + iw] *
                                       kernel[oc*in_channels*kernel_size*kernel_size + ic*kernel_size*kernel_size + kh*kernel_size + kw];
                            }
                        }
                    }
                    output[b*out_channels*(height-kernel_size+1)*(width-kernel_size+1) +
                           oc*(height-kernel_size+1)*(width-kernel_size+1) + h*(width-kernel_size+1) + w] = sum;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, output,
                  local_batch*out_channels*(height-kernel_size+1)*(width-kernel_size+1),
                  MPI_FLOAT, MPI_COMM_WORLD);
}