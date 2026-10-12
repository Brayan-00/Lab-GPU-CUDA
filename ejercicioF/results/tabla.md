Mediana de 5 corridas por version.

| Versión | Tiempo total (s) | Pasos/s | Speedup vs CPU | Speedup vs anterior | Error máximo (alturas) | Error máximo (píxel) |
|---|---:|---:|---:|---:|---:|---:|
| CPU base | 71.5369 | 12.5809 | 1.00x | - | 0 | 0 |
| GPU base | 14.3385 | 62.7679 | 4.99x | - | 0.000e+00 | 1 |
| Optimización 1 (menos datos al host) | 14.0766 | 63.936 | 5.08x | 1.02x | 0.000e+00 | 1 |
| Optimización 2 (memoria compartida) | 14.2823 | 63.015 | 5.01x | 0.99x | 0.000e+00 | 1 |
| Optimización 3 (aritmética del render) | 14.043 | 64.0889 | 5.09x | 1.02x | 0.000e+00 | 1 |
| Optimización 4 (streams y doble buffer) | 13.3711 | 67.3094 | 5.35x | 1.05x | 0.000e+00 | 1 |

Tiempo acumulado por etapa (s), mismas corridas.

| Versión | Kernel simulación | Kernel render | Copia D2H | writer.write | Espera por la GPU |
|---|---:|---:|---:|---:|---:|
| GPU base | 0.457058 | 0.622729 | 0.62003 | 12.3394 | - |
| Optimización 1 (menos datos al host) | 0.415867 | 0.54847 | 0.264801 | 12.6054 | - |
| Optimización 2 (memoria compartida) | 0.554945 | 0.674393 | 0.263106 | 12.5463 | - |
| Optimización 3 (aritmética del render) | 0.55406 | 0.439092 | 0.265028 | 12.5426 | - |
| Optimización 4 (streams y doble buffer) | 0.581768 | 0.442478 | 0.0417427 | 13.0454 | 0.00445759 |
