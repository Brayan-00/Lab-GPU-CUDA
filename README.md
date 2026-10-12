# Laboratorio Capítulo 3

## Integrantes, curso y profesor

### Integrantes

1. Fiorela Chavarría Castrillo
2. Fabián Parreaguirre Hidalgo
3. Brayan Rodríguez Villalobos

### Curso

Introducción a la Computación Heterogénea - EL-5859

### Profesor

Dr. Luis G. León-Vega

### Características del sistema



## Ejercicio A

``` bash
cd drop-simulation
meson setup build
```
Tal como se indica en las instrucciones se intenta ejecutar `meson setup -C` lo cual retorna un error. En su lugar se ejecuta:

``` bash 
ninja -C build
```
Esto retorna el siguiente error:

``` bash
ninja: Entering directory `build'
[1/2] Compiling C++ object 'drop_simulation@exe/src_main.cpp.o'.
FAILED: drop_simulation@exe/src_main.cpp.o
c++  -Idrop_simulation@exe -I. -I.. -I/usr/include/opencv4/opencv -I/usr/include/opencv4 -fdiagnostics-color=always -pipe -D_FILE_OFFSET_BITS=64 -Wall -Winvalid-pch -Wnon-virtual-dtor -Wextra -Wpedantic -std=c++17 -O3 -MD -MQ 'drop_simulation@exe/src_main.cpp.o' -MF 'drop_simulation@exe/src_main.cpp.o.d' -o 'drop_simulation@exe/src_main.cpp.o' -c ../src/main.cpp
../src/main.cpp:4:10: fatal error: filesystem: No such file or directory
 #include <filesystem>
          ^~~~~~~~~~~~
compilation terminated.
ninja: build stopped: subcommand failed.
```

Por lo que se realizan cambios en el programa para poder ejecutarlo, específicamente los problemas relacionados a las versiones de c++ soportadas por la plataforma Jetson Nano.

Después de ejecutar el programa se obtiene el siguiente resultado:

``` bash
Frame 0 / 60
Frame 6 / 60
Frame 12 / 60
Frame 18 / 60
Frame 24 / 60
Frame 30 / 60
Frame 36 / 60
Frame 42 / 60
Frame 48 / 60
Frame 54 / 60
Tiempo simulate_step: 0.391052 s
Tiempo render_frame: 5.07178 s
Tiempo writer.write: 1.04145 s
Porcentaje simulate_step: 6.01111 %
Porcentaje render_frame: 77.9616 %
Porcentaje writer.write: 16.0088 %
Video generado: output/drop_simulation.mp4
Tiempo: 6.50548 s
Pasos simulados: 60
Rendimiento: 9.22299 pasos/s
```



## Ejercicio B



## Ejercicio C

Para realizar el perfilado del programa se utilizaron dos estrategias distintas: Instrumentación manual con temporizadores y Google Performance Tools. Algunos de los motivos por los que se eligieron estas dos, es porque `perf` no estaba disponible y porque callgrind demostró tardar bastante tiempo.

### Instrumentación manual con temporizadores

Utilizando la biblioteca std::chronos se realizaron mediciones de tiempo en la ejecución del programa y se obtuvieron los siguientes datos:

``` bash
Tiempo simulate_step: 0.391052 s
Tiempo render_frame: 5.07178 s
Tiempo writer.write: 1.04145 s
Porcentaje simulate_step: 6.01111 %
Porcentaje render_frame: 77.9616 %
Porcentaje writer.write: 16.0088 %
Video generado: output/drop_simulation.mp4
Tiempo: 6.50548 s
Pasos simulados: 60
Rendimiento: 9.22299 pasos/s
```


### Google Performance Tools

Se ejecutaron los siguientes comandos para utilizar la herramienta:

```bash 
 CPUPROFILE=cpu_profile.prof LD_PRELOAD="/home/el5859/local-tools/libgoogle-perftools4/usr/lib/aarch64-linux-gnu/libprofiler.so.0.4.8" ./build/drop_simulation

 ~/local-tools/google-perftools/usr/bin/google-pprof --text ./build/drop_simulation ./cpu_profile.prof > cpu_profiling_gperftools.txt

```

Lo que da como resultado:

``` bash

Total: 658 samples
     302  45.9%  45.9%      304  46.2% main
     153  23.3%  69.1%      153  23.3% vrndn_f64 (inline)
      40   6.1%  75.2%      237  36.0% __powf
      35   5.3%  80.5%       35   5.3% av_jni_get_java_vm
      30   4.6%  85.1%       30   4.6% log2_inline (inline)
      25   3.8%  88.9%       28   4.3% avpriv_mpa_decode_header
      18   2.7%  91.6%       18   2.7% sws_get_class
      16   2.4%  94.1%       16   2.4% av_parser_close
      14   2.1%  96.2%      167  25.4% exp2_inline (inline)
       4   0.6%  96.8%        4   0.6% _init
       4   0.6%  97.4%        4   0.6% avpriv_aac_parse_header
       3   0.5%  97.9%        3   0.5% __GI_memset
       3   0.5%  98.3%        3   0.5% av_picture_pad
       2   0.3%  98.6%        2   0.3% __memcpy_generic
       2   0.3%  98.9%        2   0.3% av_mediacodec_release_buffer
       2   0.3%  99.2%        2   0.3% avpriv_copy_pce_data
       1   0.2%  99.4%        1   0.2% __GI___expf
       1   0.2%  99.5%        1   0.2% av_dct_end
       1   0.2%  99.7%        1   0.2% av_picture_copy
       1   0.2%  99.8%        1   0.2% avcodec_default_get_buffer2
       1   0.2% 100.0%        1   0.2% avpriv_fits_header_parse_line
       0   0.0% 100.0%      304  46.2% __libc_start_main
       0   0.0% 100.0%      304  46.2% _start
       0   0.0% 100.0%      153  23.3% roundtoint (inline)

```



