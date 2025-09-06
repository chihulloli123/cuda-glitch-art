#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;

// Simple CUDA kernel: swap red/blue channels for glitch effect
__global__ void glitchKernel(unsigned char* img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        // Swap Red and Blue channels (glitch)
        unsigned char temp = img[idx];
        img[idx] = img[idx + 2];
        img[idx + 2] = temp;
    }
}

int main() {
    // Load your tattoo image
    Mat input = imread("src/tattoo_portal.jpg", IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Could not load tattoo image!\n";
        return -1;
    }

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imgSize = width * height * channels;

    // Allocate GPU memory
    unsigned char *d_img;
    cudaMalloc((void**)&d_img, imgSize);

    // Copy image to GPU
    cudaMemcpy(d_img, input.data, imgSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    glitchKernel<<<grid, block>>>(d_img, width, height, channels);
    cudaDeviceSynchronize();

    // Copy back to CPU
    Mat output(height, width, CV_8UC3);
    cudaMemcpy(output.data, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Save result
    imwrite("output/glitch_portal.jpg", output);

    cudaFree(d_img);

    std::cout << "✅ Glitch image saved to output/glitch_portal.jpg\n";
    return 0;
}
