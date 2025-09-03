#include <iostream>
#include <fstream>
#include <complex>

int main() {
    const int width = 800;
    const int height = 600;
    const int max_iter = 1000;

    std::ofstream img("../output/fractal.ppm");
    img << "P3\n" << width << " " << height << " 255\n";

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::complex<double> c(
                (x - width/2.0) * 4.0/width,
                (y - height/2.0) * 4.0/width
            );
            std::complex<double> z = 0;
            int iter = 0;
            while (abs(z) < 2.0 && iter < max_iter) {
                z = z*z + c;
                iter++;
            }
            int color = (iter * 255) / max_iter;
            img << color << " " << 0 << " " << (255-color) << " ";
        }
        img << "\n";
    }
    img.close();
    std::cout << "Fractal image saved to ../output/fractal.ppm\n";
    return 0;
}

// :) ... seeing that makes my little heart so happy i could cry <3 
// chi pisa 