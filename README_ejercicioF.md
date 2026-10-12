# Ejercicio F: optimización de la versión GPU

Introducción a la Computación Heterogénea - EL-5859. Todo el código, el script de medición y los resultados están en la carpeta [`ejercicioF/`](ejercicioF/); este archivo lo explica.

```text
ejercicioF/
├── meson.build           compila las 5 versiones GPU y la referencia en CPU
├── run_benchmarks.sh     corre todas las versiones y arma results/tabla.md
├── src/
│   ├── common.h          configuración, referencia en CPU, temporizadores y validación
│   ├── v0_base.cu        GPU base
│   ├── v1_menos_datos.cu
│   ├── v2_shared_memory.cu
│   ├── v3_aritmetica.cu
│   ├── v4_async.cu       versión final
│   └── cpu_base.cpp      copia sin cambios de drop-simulation/src/main_base.cpp
├── results/              corrida final (relojes fijos, 5 repeticiones)
└── results_corrida1/     primera corrida (sin fijar relojes, con el writer en gris)
```
## Versiones y optimizaciones (Ejercicio F)

Las optimizaciones se hicieron en la Jetson Nano y cada una parte de la anterior. Los archivos están en `ejercicioF/src/`:

| Versión | Archivo | Qué cambia |
|---|---|---|
| GPU base | `v0_base.cu` | Es `ported.cu` con temporizadores y `--validate`. Los kernels no cambian. |
| Optimización 1 | `v1_menos_datos.cu` | El render escribe gris (1 byte) en vez de BGR (3 bytes), se quita la sincronización de cada paso y la CPU pasa el gris a BGR antes de escribir. |
| Optimización 2 | `v2_shared_memory.cu` | Bloques de 32x8 y memoria compartida para los vecinos del stencil. |
| Optimización 3 | `v3_aritmetica.cu` | `powf(x, 24)` se cambia por multiplicaciones. |
| Optimización 4 | `v4_async.cu` | Stream propio, copia asíncrona y doble buffer: la GPU trabaja en el cuadro N+1 mientras la CPU codifica el N. |

`common.h` tiene lo que comparten todas las versiones: configuración, referencia en CPU, temporizadores y validación.

### Compilar y ejecutar

```bash
cd ejercicioF
meson setup build
ninja -C build
./build/drop_simulation_v4_async              # genera output/v4_async.mp4
./build/drop_simulation_v4_async --validate   # compara contra la CPU
./run_benchmarks.sh                           # corre todas las versiones y arma results/tabla.md
```

Cada ejecución imprime el tiempo total, los pasos por segundo y el tiempo acumulado de cada etapa. Los kernels y la copia se miden con `cudaEvent` y `writer.write` con `std::chrono`. El perfilado con nvprof queda en `ejercicioF/results/*_nvprof.txt`.

### Optimización 1: menos datos hacia el host

**Problema.** Los tres canales de cada cuadro son iguales, pero se copiaban 1.2 MB por cuadro. Además el programa esperaba a la GPU después de cada paso con `cudaDeviceSynchronize`, aunque los kernels de un mismo stream ya se ejecutan en orden.

**Cambio.**

```cpp
// v0
image[3 * idx] = gray;
image[3 * idx + 1] = gray;
image[3 * idx + 2] = gray;

// v1
image[idx] = static_cast<unsigned char>(intensity);
```

Se quitó `cudaDeviceSynchronize()` de `step()`, se reutilizan los `cv::Mat` del host y la copia baja de 1,228,800 a 409,600 bytes por cuadro. Antes de escribir, la CPU convierte el gris a BGR con `cv::cvtColor`.

**Primer intento.** La primera versión abría el `VideoWriter` con `isColor=false` y le pasaba el gris directo. Con eso la copia sí bajó (2.10 s a 0.90 s en la primera corrida), pero `writer.write` subió de 19.05 s a 25.86 s y el total empeoró de 31.70 s a 36.81 s. Se descartó y se dejó el writer en color, que es lo que muestran las tablas de abajo. Las salidas de esa primera corrida están en `ejercicioF/results_corrida1/`.

**Evidencia.** En el nvprof de la base la copia `memcpy DtoH` tarda 123 µs por cuadro, el 10 % del tiempo de GPU, y el programa esperaba a la GPU en cada paso.

### Optimización 2: acceso a memoria en los kernels

**Problema.** El stencil de 5 puntos lee cada altura cinco veces de memoria global (la celda y sus 4 vecinos), y el render hace lo mismo. Con bloques de 16x16 cada fila del warp lee solo 64 bytes seguidos.

**Cambio.** Bloques de 32x8 hilos y un `tile` en memoria compartida que cada bloque llena una sola vez (su pedazo más una celda de borde):

```cpp
__shared__ float tile[kTileH][kTileW];   // 10 x 34 floats
load_tile(current, tile, width, height, x0, y0);
__syncthreads();

const float laplacian = tile[ty][tx - 1] + tile[ty][tx + 1] +
                        tile[ty - 1][tx] + tile[ty + 1][tx] - 4.0f * center;
```

El render usa el mismo `tile`. En los bordes de la malla se repite el valor del borde, igual que el render original.

**Evidencia.** En el nvprof de la versión 1 los dos kernels tardan 446 µs (simulación) y 639 µs (render) por cuadro, y los dos hacen casi todo en lecturas de memoria. La hipótesis era que leer cada valor una sola vez los acortaría. Resultado en el Ejercicio G.

### Optimización 3: aritmética del render

**Problema.** El render calcula `powf(diffuse, 24.0f)` para cada pixel. En el perfil de CPU con gperftools `__powf` (con `exp2` y `log2`) sumaba 36 % acumulado, y en GPU `powf` también usa `log2`, `exp2` y el manejo de casos especiales.

**Cambio.** El exponente es un entero constante, así que se calcula con cinco multiplicaciones:

```cpp
// x^24 = x^16 * x^8
__device__ float pow24(float x) {
    const float x2 = x * x;
    const float x4 = x2 * x2;
    const float x8 = x4 * x4;
    const float x16 = x8 * x8;
    return x16 * x8;
}
```

No se usó `-use_fast_math`: sin `powf` casi no quedan funciones especiales en el kernel de render, y la bandera también habría cambiado la simulación.

**Evidencia.** En el nvprof de la base el render es el kernel más largo (667 µs por cuadro, 54 % del tiempo de GPU).

### Optimización 4: streams y doble buffer

**Problema.** El tiempo de un cuadro era la suma de la GPU (simulación, render y copia) y la CPU (codificar el video), porque cada una esperaba a la otra.

**Cambio.** Todo el trabajo de GPU va en un stream propio, la copia es `cudaMemcpyAsync` hacia memoria pinned y hay dos buffers en el host. La GPU deja en cola el cuadro N+1 antes de que la CPU codifique el N:

```cpp
enqueue_frame(0);
for (int frame = 0; frame < frames; ++frame) {
    if (frame + 1 < frames) {
        enqueue_frame(frame + 1);   // la GPU sigue con el siguiente
    }
    slot.copy.wait();               // espera solo a su propio cuadro
    writer.write(image);            // la CPU codifica mientras tanto
}
```

```text
v3 (secuencial)   GPU: [cuadro N]            [cuadro N+1]
                  CPU:           [codificar N]             [codificar N+1]

v4 (doble buffer) GPU: [cuadro N+1]          [cuadro N+2]
                  CPU: [codificar N ......]  [codificar N+1 ......]
```

**Evidencia.** En la versión 3 `writer.write` son 12.5 s de 14.0 s (89 %), mientras que los kernels y la copia suman 1.2 ms por cuadro. Mientras la CPU codifica, la GPU no tiene nada que hacer.


## Resultados (Ejercicio G)

Todo se midió en la Jetson Nano con el video completo (30 s, 900 pasos), con `sudo nvpmodel -m 0` y `sudo jetson_clocks`. Los tiempos son la mediana de 5 corridas y salen de `results/tabla.md`, que arma `run_benchmarks.sh`.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
ninja -C build
sudo env REPS=5 ./run_benchmarks.sh
```

| | |
|---|---|
| Equipo | Jetson Nano (GPU Maxwell, `sm_53`) |
| Sistema | Ubuntu 18.04.6 LTS, kernel 4.9.253-tegra, aarch64 |
| Herramientas | gcc 7.5.0, nvcc, meson 0.45.1, ninja 1.8.2 |
| CUDA y OpenCV | _pendiente_ |
| Perfilado | nvprof (150 cuadros por versión) |

### Resultados

| Versión | Tiempo total (s) | Pasos/s | Speedup vs CPU | Speedup vs anterior | Error máximo |
|---|---:|---:|---:|---:|---:|
| CPU base | 71.54 | 12.58 | 1.00x | - | 0 |
| GPU base | 14.34 | 62.77 | 4.99x | - | 0 |
| Optimización 1 | 14.08 | 63.94 | 5.08x | 1.02x | 0 |
| Optimización 2 | 14.28 | 63.02 | 5.01x | 0.99x | 0 |
| Optimización 3 | 14.04 | 64.09 | 5.09x | 1.02x | 0 |
| Optimización 4 | 13.37 | 67.31 | 5.35x | 1.05x | 0 |

El error máximo es la diferencia absoluta más grande en la malla de alturas contra la CPU después de 900 pasos, medida con `--validate` (salidas en `results/*_validacion.txt`). En píxeles el máximo es 1 de 255 en todas las versiones.

Tiempo por llamada de cada kernel y de la copia, según nvprof (µs):

| Versión | Simulación | Render | Copia D2H |
|---|---:|---:|---:|
| GPU base | 429 | 667 | 123 |
| Optimización 1 | 446 | 639 | 42 |
| Optimización 2 | 612 | 777 | 42 |
| Optimización 3 | 613 | 520 | 42 |
| Optimización 4 | 618 | 530 | 44 |

La copia host a device ocurre una sola vez al inicio (2 llamadas, menos de 1 ms en total en todas las versiones).

Tiempo acumulado por etapa en el video completo (s):

| Versión | Kernel simulación | Kernel render | Copia D2H | writer.write | Espera por la GPU |
|---|---:|---:|---:|---:|---:|
| GPU base | 0.46 | 0.62 | 0.62 | 12.34 | - |
| Optimización 1 | 0.42 | 0.55 | 0.26 | 12.61 | - |
| Optimización 2 | 0.55 | 0.67 | 0.26 | 12.55 | - |
| Optimización 3 | 0.55 | 0.44 | 0.27 | 12.54 | - |
| Optimización 4 | 0.58 | 0.44 | 0.04 | 13.05 | 0.004 |

Los kernels y la copia se miden con `cudaEvent` y `writer.write` incluye `putText`, la conversión a BGR y la codificación. Los tiempos de nvprof son los más fiables para comparar kernels.

### Resultados de cada optimización

| Optimización | Total | Cambio en la etapa | Error alturas / píxel |
|---|---|---|---|
| 1. Menos datos al host | 14.34 → 14.08 s (1.02x) | Copia D2H de 123 a 42 µs. `writer.write` sube 0.27 s por la conversión a BGR. | 0 / 1 |
| 2. Memoria compartida | 14.08 → 14.28 s (0.99x) | Simulación de 446 a 612 µs y render de 639 a 777 µs. Empeoró. | 0 / 1 |
| 3. Aritmética del render | 14.28 → 14.04 s (1.02x) | Render de 777 a 520 µs (33 % menos). | 0 / 1 |
| 4. Streams y doble buffer | 14.04 → 13.37 s (1.05x) | Espera por la GPU de solo 0.004 s. `writer.write` sube 0.5 s. | 0 / 1 |

Ninguna cambió la precisión: el único cambio aritmético (`x^24` con multiplicaciones) no movió el error en píxeles.

### Salidas de nvprof

Base (`results/v0_base_nvprof.txt`):

```text
 GPU activities:   54.48%  100.04ms  150  666.91us  render_frame_kernel(...)
                   35.01%  64.290ms  150  428.60us  simulate_step_kernel(...)
                   10.04%  18.436ms  150  122.91us  [CUDA memcpy DtoH]
```

Versión final (`results/v4_async_nvprof.txt`):

```text
 GPU activities:   51.63%  92.686ms  150  617.90us  simulate_step_kernel(...)
                   44.29%  79.515ms  150  530.10us  render_frame_kernel(...)
                    3.70%  6.6415ms  150  44.276us  [CUDA memcpy DtoH]
```

### Análisis de rendimiento

- **El video es el límite.** En la base `writer.write` son 12.3 s de 14.3 s (86 %), y todo lo de GPU (kernels y copia) suma unos 1.2 ms por cuadro, cerca del 8 %. Por eso ninguna optimización de GPU podía dar más de 1.08x en total (ley de Amdahl) y la versión final llegó a 1.07x.
- **Optimización 1.** La copia se triplicó en velocidad, pero lo ganado es poco en total y parte se pierde en convertir el gris a BGR en la CPU. El primer intento, pasar el gris directo al `VideoWriter`, fue peor: en la primera corrida `writer.write` subió de 19.05 s a 25.86 s.
- **Optimización 2.** Salió peor en las dos corridas: en la primera, simulación de 3.67 ms a 6.29 ms por llamada, y en la segunda de 446 a 612 µs. Probablemente llenar el tile (división, módulo y límites por cada elemento, más la barrera) cuesta más que lo que ahorra, porque en la Nano la caché de lectura ya aprovecha el reuso del stencil. No se aisló qué parte del cambio causa la pérdida.
- **Optimización 3.** Fue la que más movió un kernel: el render baja un tercio con solo cambiar `powf` por multiplicaciones.
- **Optimización 4.** La GPU queda escondida detrás del encoder (espera de 0.004 s), pero `writer.write` sube 0.5 s, posiblemente porque CPU y GPU comparten la misma RAM y compiten por ella.
- **Relojes.** Una primera corrida sin fijar relojes (`results_corrida1/`) dio kernels hasta 12 veces más lentos (render de la base: 7.83 ms por llamada contra 0.67 ms) y mucha variación, y la CPU tardó 112 s en vez de 71.5 s. Por eso todas las medidas finales usan `nvpmodel -m 0` y `jetson_clocks`.

### Lectura previa

- **Acceso a memoria.** El stencil de 5 puntos reutiliza cada valor cinco veces. Se intentó aprovecharlo con memoria compartida (optimización 2) y empeoró: reducir lecturas globales no ayuda si agregar el tile cuesta más que ellas.
- **Precisión contra velocidad.** Todo se mantiene en `float32`. El único cambio aritmético (optimización 3) no usa aproximaciones del compilador y no cambió el error.
- **Comunicación host-device.** Las optimizaciones 1 y 4 no cambian lo que calcula ningún kernel, solo cuántos datos se mueven y cuándo, y aun así cambian el tiempo total.

### Conclusiones

La versión final es 5.35x más rápida que la CPU (71.5 s a 13.4 s) y 1.07x más rápida que la primera versión GPU. Casi todo el salto viene del port a GPU; las cuatro optimizaciones sumaron poco porque el tiempo restante es del encoder de video en CPU, que ya no depende de la GPU.

De las cuatro, la aritmética del render fue la más efectiva a nivel de kernel y los streams con doble buffer la que más bajó el total. La memoria compartida empeoró y el primer intento de la optimización 1 también, y se descubrió porque se midió antes y después de cada cambio. Para ganar más habría que atacar el encoder, por ejemplo con el codificador por hardware de la Jetson.
