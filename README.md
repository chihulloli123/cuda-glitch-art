# cuda-glitch-art
Tattoo created by **Micah Hutson**  
Created @ Electric Age Tattoo Company (Sherman, TX)  
Owner of Spirit Tattoo Club (Tulsa, OK)  

## CUDA Glitch + Fractal Art
A CUDA-powered art generator that explores the intersection of **math, art, and GPU acceleration**.
- Mandelbrot/Julia fractals (CPU + GPU accelerated)  
- Tattoo-inspired color mapping (red/orange spirals, dark edge)  
- Benchmarks: CPU vs GPU runtime  
- Extensible: add glitch filters or other math-art generators  

## Getting Started

### CPU Version
```bash
g++ fractal_cpu.cpp -o fractal_cpu
./fractal_cpu

cuda-glitch-art

Tattoo created by Micah Hutson

- Created @ Electric Age Tattoo Company (Sherman, TX)

- Owner of Spirit Tattoo Club (Tulsa, OK)

Overview

cuda-glitch-art is a CUDA-powered art generator that explores the intersection of math, art, and GPU acceleration.

- Mandelbrot/Julia fractals (CPU + GPU accelerated)

- Tattoo-inspired color mapping (spirals, dark edges)

- Benchmarks: CPU vs GPU runtime

- Extensible: add glitch filters or other math-art generators

Requirements

Before running, make sure these are installed:

C++ Compiler (g++)

Download MinGW-w64
 → extract to C:\mingw64

Add C:\mingw64\bin to PATH

Verify with:

g++ --version


ImageMagick (for converting .ppm to .png)

Download

Add its install dir (e.g. C:\Program Files\ImageMagick-7.1.1-Q16-HDRI) to PATH

Verify with:

magick -version


CUDA Toolkit (optional, for GPU fractals)

Download

Verify with:

nvcc --version

Getting Started

Clone, build, and run:

# Clone the repo
git clone https://github.com/chihulloli123/cuda-glitch-art
cd cuda-glitch-art

# Build CPU fractal generator
g++ src/fractal_cpu.cpp -o fractal_cpu

# Run (outputs fractal.ppm)
./fractal_cpu

# Convert to PNG with ImageMagick
magick output/fractal.ppm output/fractal.png


Now you’ll have a fractal image saved in output/.
(Add CPU fractal generator and update README with requirements + getting started)
