#!/bin/bash
# Mide todas las versiones y arma las tablas del ejercicio G en results/tabla.md.
# Se corre desde drop-simulation/ despues de compilar con meson:
#
#   ./run_benchmarks.sh
#   REPS=5 ./run_benchmarks.sh       corridas por version (3 por defecto)
#   SKIP_CPU=1 ./run_benchmarks.sh   no repite la version CPU (tarda varios minutos)

# sudo no hereda el PATH del usuario y nvprof vive aqui
PATH="$PATH:/usr/local/cuda/bin"

BUILD=build
OUT=results
REPS=${REPS:-3}
VERSIONS="v0_base v1_menos_datos v2_shared_memory v3_aritmetica v4_async"

mkdir -p "$OUT"

# Numero de una linea "Clave: numero unidad" dentro de un archivo.
valor() {
    grep "^$2:" "$1" | head -1 | cut -d: -f2 | awk '{print $1}'
}

# Archivo de la corrida con el tiempo mediano de una version.
mediana() {
    for f in "$OUT/$1"_run*.txt; do
        echo "$(valor "$f" Tiempo) $f"
    done | sort -n | sed -n "$(( (REPS + 1) / 2 ))p" | cut -d' ' -f2
}

nombre() {
    case "$1" in
        v0_base)          echo "GPU base" ;;
        v1_menos_datos)   echo "Optimización 1 (menos datos al host)" ;;
        v2_shared_memory) echo "Optimización 2 (memoria compartida)" ;;
        v3_aritmetica)    echo "Optimización 3 (aritmética del render)" ;;
        v4_async)         echo "Optimización 4 (streams y doble buffer)" ;;
    esac
}

# ---------------------------------------------------------------- corridas
if [ -z "${SKIP_CPU:-}" ]; then
    echo "== CPU base"
    "./$BUILD/drop_simulation_cpu" > "$OUT/cpu_base.txt"
fi

for v in $VERSIONS; do
    echo "== $v"
    for i in $(seq 1 "$REPS"); do
        "./$BUILD/drop_simulation_$v" > "$OUT/${v}_run$i.txt" || exit 1
    done
    "./$BUILD/drop_simulation_$v" --validate > "$OUT/${v}_validacion.txt" || exit 1
done

# ---------------------------------------------------------------- perfilado
# Una corrida corta (5 s de video) por version, para no esperar de mas.
for v in $VERSIONS; do
    if command -v nvprof > /dev/null; then
        nvprof --print-gpu-summary "./$BUILD/drop_simulation_$v" --seconds 5 \
            --output "output/perfil_$v.mp4" > /dev/null 2> "$OUT/${v}_nvprof.txt"
    elif command -v nsys > /dev/null; then
        nsys profile --stats=true --force-overwrite=true -o "$OUT/${v}_nsys" \
            "./$BUILD/drop_simulation_$v" --seconds 5 --output "output/perfil_$v.mp4" \
            > "$OUT/${v}_nsys.txt"
    else
        echo "No hay nvprof ni nsys, se omite el perfilado de $v"
    fi
done

# ---------------------------------------------------------------- tablas
cpu_tiempo=$(valor "$OUT/cpu_base.txt" Tiempo)
cpu_pasos=$(valor "$OUT/cpu_base.txt" Rendimiento)

{
    echo "Mediana de $REPS corridas por version."
    echo
    echo "| Versión | Tiempo total (s) | Pasos/s | Speedup vs CPU | Speedup vs anterior | Error máximo (alturas) | Error máximo (píxel) |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    echo "| CPU base | $cpu_tiempo | $cpu_pasos | 1.00x | - | 0 | 0 |"

    anterior=$cpu_tiempo
    for v in $VERSIONS; do
        f=$(mediana "$v")
        t=$(valor "$f" Tiempo)
        p=$(valor "$f" Rendimiento)
        err=$(valor "$OUT/${v}_validacion.txt" "Error maximo alturas")
        pix=$(valor "$OUT/${v}_validacion.txt" "Error maximo pixel")
        vs_cpu=$(awk -v a="$cpu_tiempo" -v b="$t" 'BEGIN { printf "%.2fx", a / b }')
        vs_ant=$(awk -v a="$anterior" -v b="$t" 'BEGIN { printf "%.2fx", a / b }')
        [ "$v" = "v0_base" ] && vs_ant="-"
        echo "| $(nombre "$v") | $t | $p | $vs_cpu | $vs_ant | $err | $pix |"
        anterior=$t
    done

    echo
    echo "Tiempo acumulado por etapa (s), mismas corridas."
    echo
    echo "| Versión | Kernel simulación | Kernel render | Copia D2H | writer.write | Espera por la GPU |"
    echo "|---|---:|---:|---:|---:|---:|"
    for v in $VERSIONS; do
        f=$(mediana "$v")
        espera=$(valor "$f" "Espera por la GPU")
        echo "| $(nombre "$v") | $(valor "$f" "Tiempo kernel simulacion") | $(valor "$f" "Tiempo kernel render") | $(valor "$f" "Tiempo copia D2H") | $(valor "$f" "Tiempo writer.write") | ${espera:--} |"
    done
} > "$OUT/tabla.md"

cat "$OUT/tabla.md"
