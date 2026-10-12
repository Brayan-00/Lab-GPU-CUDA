project(
  'ejercicioF',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv_dep = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

# Rutas de CUDA configuradas a mano, igual que en drop-simulation
cuda_inc = include_directories('/usr/local/cuda/include')
cuda_dep = declare_dependency(
  include_directories: cuda_inc,
  link_args: ['-L/usr/local/cuda/lib64', '-lcudart']
)

nvcc = find_program('nvcc', required: true)

# Versiones GPU. Cada una parte de la anterior (ver el comentario al inicio
# de cada archivo). v0 es la base GPU con mediciones.
gpu_versions = [
  'v0_base',
  'v1_menos_datos',
  'v2_shared_memory',
  'v3_aritmetica',
  'v4_async',
]

foreach name : gpu_versions
  version_obj = custom_target(
    name + '_obj',
    input: 'src/' + name + '.cu',
    output: name + '.o',
    depend_files: ['src/common.h'],
    command: [
      nvcc,
      '-c', '@INPUT@',
      '-o', '@OUTPUT@',
      '-O3',
      '-std=c++14',
      '-lineinfo',
      '-I/usr/local/cuda/include',
      '-I/usr/include/opencv4',
      '-gencode', 'arch=compute_53,code=sm_53'
    ]
  )

  executable(
    'drop_simulation_' + name,
    sources: [version_obj],
    dependencies: [opencv_dep, cuda_dep],
    install: false,
  )
endforeach

# Referencia en CPU (copia sin cambios de drop-simulation/src/main_base.cpp)
executable(
  'drop_simulation_cpu',
  sources: ['src/cpu_base.cpp'],
  dependencies: [opencv_dep],
  install: false,
)
