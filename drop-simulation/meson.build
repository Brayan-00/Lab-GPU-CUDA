project(
  'drop-simulation',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv_dep = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

# Configuración manual de rutas de CUDA
cuda_inc = include_directories('/usr/local/cuda/include')
cuda_dep = declare_dependency(
  include_directories: cuda_inc,
  link_args: ['-L/usr/local/cuda/lib64', '-lcudart']
)

nvcc = find_program('nvcc', required: true)

# Compilar src/ported.cu directamente a un archivo objeto con nvcc
cuda_obj = custom_target(
  'ported_cuda_obj',
  input: 'src/ported.cu',
  output: 'ported.o',
  command: [
    nvcc,
    '-c', '@INPUT@',
    '-o', '@OUTPUT@',
    '-O3',
    '-std=c++14',
    '-I/usr/local/cuda/include',
    '-I/usr/include/opencv4',
    '-gencode', 'arch=compute_53,code=sm_53'
  ]
)

# Ejecutable unificado para la versión GPU (usando el main dentro de ported.cu)
executable(
  'drop_simulation_gpu',
  sources: [cuda_obj],
  dependencies: [opencv_dep, cuda_dep],
  install: false,

)

# 2. Ejecutable para la versión de CPU
executable(
  'drop_simulation_cpu',
  sources: ['src/main_base.cpp'],
  dependencies: [opencv_dep],
  install: false,
)
