Mediana de 3 corridas por version.

| Versión | Tiempo total (s) | Pasos/s | Speedup vs CPU | Speedup vs anterior | Error máximo (alturas) | Error máximo (píxel) |
|---|---:|---:|---:|---:|---:|---:|
| CPU base | 112.196 | 8.02169 | 1.00x | - | 0 | 0 |
| GPU base | 31.6985 | 28.3925 | 3.54x | - | 0.000e+00 | 1 |
| Optimización 1 (menos datos al host) | 36.81 | 24.4499 | 3.05x | 0.86x | 0.000e+00 | 1 |
| Optimización 2 (memoria compartida) | 40.9001 | 22.0049 | 2.74x | 0.90x | 0.000e+00 | 1 |
| Optimización 3 (aritmética del render) | 32.4831 | 27.7068 | 3.45x | 1.26x | 0.000e+00 | 1 |
| Optimización 4 (streams y doble buffer) | 26.7878 | 33.5974 | 4.19x | 1.21x | 0.000e+00 | 1 |

Tiempo acumulado por etapa (s), mismas corridas.

| Versión | Kernel simulación | Kernel render | Copia D2H | writer.write | Espera por la GPU |
|---|---:|---:|---:|---:|---:|
| GPU base | 3.34268 | 6.80169 | 2.09531 | 19.0504 | - |
| Optimización 1 (menos datos al host) | 3.10677 | 6.53169 | 0.902917 | 25.86 | - |
| Optimización 2 (memoria compartida) | 5.51504 | 7.97573 | 0.926433 | 26.0606 | - |
| Optimización 3 (aritmética del render) | 2.83011 | 2.55111 | 1.0291 | 25.7935 | - |
| Optimización 4 (streams y doble buffer) | 2.74962 | 2.52364 | 0.176366 | 26.3465 | 0.0109508 |
