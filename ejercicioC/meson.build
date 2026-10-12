project(
  'ejercicioC',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv_dep = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

# main.cpp: versión de 2 s de video con temporizadores (la que se perfiló con gperftools)
executable(
  'drop_simulation',
  sources: ['src/main.cpp'],
  dependencies: [opencv_dep],
  install: false,
)

# main_timers.cpp: igual, con los 30 s de video completos.
# main_callgrind.cpp es idéntico a este archivo.
executable(
  'drop_simulation_timers',
  sources: ['src/main_timers.cpp'],
  dependencies: [opencv_dep],
  install: false,
)
