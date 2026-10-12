#!/bin/bash

echo "=========================================="
echo "   INICIANDO COMPARACIÓN CPU vs GPU       "
echo "=========================================="

# 1. Ejecutar versión CPU
echo "[1/3] Ejecutando versión CPU..."
./build/drop_simulation_cpu
CPU_EXIT_CODE=$?

if [ $CPU_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] La simulación de CPU falló."
    exit 1
fi

# 2. Ejecutar versión GPU
echo "[2/3] Ejecutando versión GPU..."
./build/drop_simulation_gpu
GPU_EXIT_CODE=$?

if [ $GPU_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] La simulación de GPU falló."
    exit 1
fi

# 3. Comparar los resultados usando un script ligero en Python
echo "[3/3] Calculando métricas de error (MAE)..."
python3 -c "
import numpy as np

# Cargar las mallas binarias generadas por ambos programas
cpu_grid = np.fromfile('output/final_grid_cpu.bin', dtype=np.float32)
gpu_grid = np.fromfile('output/final_grid_gpu.bin', dtype=np.float32)

if cpu_grid.size != gpu_grid.size:
    print('[ERROR] Los tamaños de las mallas no coinciden!')
    exit(1)

# Calcular MAE y Error Máximo
mae = np.mean(np.abs(cpu_grid - gpu_grid))
max_err = np.max(np.abs(cpu_grid - gpu_grid))

print('------------------------------------------')
print(f'Elementos evaluados   : {cpu_grid.size}')
print(f'Error Absoluto Medio  : {mae}')
print(f'Error Absoluto Máximo : {max_err}')
print('------------------------------------------')

if mae < 1e-4:
    print('[EXITO] Las mallas son equivalentes.')
else:
    print('[ADVERTENCIA] Las mallas difieren.')
"

echo "=========================================="
echo "   COMPARACIÓN FINALIZADA CON ÉXITO       "
echo "=========================================="
