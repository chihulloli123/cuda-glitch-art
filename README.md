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
