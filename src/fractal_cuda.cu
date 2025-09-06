@'
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 1000

__global__ void mandelbrot(unsigned char *img) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    double scale_x = 3.5 / WIDTH;
    double scale_y = 2.0 / HEIGHT;

    double cx = x * scale_x - 2.5;
    double cy = y * scale_y - 1.0;

    double zx = 0.0, zy = 0.0;
    int iter = 0;

    while (zx * zx + zy * zy < 4.0 && iter < MAX_ITER) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        iter++;
    }

    int offset = (y * WIDTH + x) * 3;

    if (iter == MAX_ITER) {
        img[offset + 0] = 0;
        img[offset + 1] = 0;
        img[offset + 2] = 0;
    } else {
        int color = (int)(255.0 * iter / MAX_ITER);
        img[offset + 0] = color;
        img[offset + 1] = 0;
        img[offset + 2] = 255 - color;
    }
}

int main() {
    size_t img_size = WIDTH * HEIGHT * 3;
    unsigned char *h_img = (unsigned char*)malloc(img_size);
    unsigned char *d_img;

    cudaMalloc((void**)&d_img, img_size);

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    mandelbrot<<<blocks, threads>>>(d_img);

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    FILE *fp = fopen("../output/fractal_cuda.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(h_img, 1, img_size, fp);
    fclose(fp);

    cudaFree(d_img);
    free(h_img);

    printf("CUDA fractal saved to ../output/fractal_cuda.ppm\n");
    return 0;
}
'@ | Set-Content fractal_cuda.cu
