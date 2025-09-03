# cuda-glitch-art
# tattoo from Micah Hutson created at Eletric Age Tattoo Company in Sherman, TX
# Spirit Tattoo Club Tulsa, OK
CUDA-powered glitch art generator — turn any image into GPU-created glitch art.
# CUDA Glitch + Fractal Art

A CUDA-powered art generator.  
- Mandelbrot/Julia fractals (CPU + GPU accelerated).  
- Tattoo-inspired color mapping (red/orange spirals, dark edge)
- Benchmarks: CPU vs GPU runtime.  
- Extensible: add glitch filters or other math-art generators.  

## Getting Started
### CPU Version
```bash
g++ fractal_cpu.cpp -o fractal_cpu
./fractal_cpu
