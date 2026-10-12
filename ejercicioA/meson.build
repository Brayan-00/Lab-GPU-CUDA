project(
  'ejercicioA',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv_dep = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

# Versión de CPU (main_base.cpp usa mkdir en vez de std::filesystem para compilar con g++ 7.5)
executable(
  'drop_simulation',
  sources: ['src/main_base.cpp'],
  dependencies: [opencv_dep],
  install: false,
)
