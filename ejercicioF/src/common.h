// Cosas que comparten las versiones GPU del ejercicio F: configuracion,
// referencia en CPU para validar, temporizadores y salida de resultados.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cuda_runtime.h>

struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation.mp4";
};

struct Options {
    bool validate = false;
};

// Uso: ./drop_simulation_vX [--validate] [--seconds S] [--output ruta.mp4]
inline Options parse_args(int argc, char** argv, Config& cfg) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--validate") {
            opt.validate = true;
        } else if (arg == "--seconds" && i + 1 < argc) {
            cfg.seconds = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output = argv[++i];
        } else {
            throw std::runtime_error("Argumento no reconocido: " + arg);
        }
    }
    return opt;
}

inline int total_frames(const Config& cfg) {
    return static_cast<int>(std::round(cfg.seconds * cfg.fps));
}

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

inline void make_parent_dirs(const std::string& path) {
    for (std::size_t pos = path.find('/'); pos != std::string::npos;
         pos = path.find('/', pos + 1)) {
        if (pos == 0) {
            continue;
        }
        const std::string dir = path.substr(0, pos);
        if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            throw std::runtime_error("No se pudo crear el directorio '" + dir +
                                     "': " + std::strerror(errno));
        }
    }
}

inline void open_writer(cv::VideoWriter& writer, const Config& cfg) {
    make_parent_dirs(cfg.output);
    writer.open(cfg.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                static_cast<double>(cfg.fps), cv::Size(cfg.width, cfg.height), true);
    if (!writer.isOpened()) {
        throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
    }
}

inline void put_label(cv::Mat& image, int frame) {
    cv::putText(image, "CUDA | frame " + std::to_string(frame), cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
}

// Mide cuanto tarda un tramo de trabajo en GPU usando dos eventos.
class EventTimer {
public:
    EventTimer() {
        cuda_check(cudaEventCreate(&start_), "cudaEventCreate");
        cuda_check(cudaEventCreate(&stop_), "cudaEventCreate");
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void begin(cudaStream_t stream = 0) { cudaEventRecord(start_, stream); }
    void end(cudaStream_t stream = 0) { cudaEventRecord(stop_, stream); }
    void wait() { cuda_check(cudaEventSynchronize(stop_), "cudaEventSynchronize"); }

    double seconds() {
        wait();
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms / 1000.0;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

using Clock = std::chrono::steady_clock;

inline double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Tiempos acumulados, en segundos.
struct Totals {
    double sim = 0.0;      // kernel de simulacion
    double render = 0.0;   // kernel de render
    double copy = 0.0;     // copia GPU -> CPU
    double encode = 0.0;   // lo que hace la CPU con el cuadro: putText, conversion a BGR y writer.write
    double wait = 0.0;     // espera del host por la GPU (solo la version async)
};

inline void print_summary(const Config& cfg, double elapsed, const Totals& t) {
    const double steps = static_cast<double>(total_frames(cfg)) * cfg.steps_per_frame;
    std::cout << "Video generado: " << cfg.output << '\n';
    std::cout << "Tiempo: " << elapsed << " s\n";
    std::cout << "Pasos simulados: " << steps << '\n';
    std::cout << "Rendimiento: " << steps / elapsed << " pasos/s\n";
    std::cout << "Tiempo kernel simulacion: " << t.sim << " s\n";
    std::cout << "Tiempo kernel render: " << t.render << " s\n";
    std::cout << "Tiempo copia D2H: " << t.copy << " s\n";
    std::cout << "Tiempo writer.write: " << t.encode << " s\n";
    if (t.wait > 0.0) {
        std::cout << "Espera por la GPU: " << t.wait << " s\n";
    }
}

// ---------------------------------------------------------------------------
// Referencia en CPU (misma logica que main_base.cpp, sin OpenCV)
// ---------------------------------------------------------------------------

inline int index_of(int x, int y, int width) {
    return y * width + x;
}

inline void add_drop(std::vector<float>& current, std::vector<float>& previous,
                     const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse = cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}

inline float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min(std::min(x, y), std::min(cfg.width - 1 - x, cfg.height - 1 - y));
    if (dist >= band) {
        return cfg.damping;
    }
    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

inline void cpu_step(const std::vector<float>& previous, const std::vector<float>& current,
                     std::vector<float>& next, const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;
    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian = current[idx - 1] + current[idx + 1] +
                                    current[idx - cfg.width] + current[idx + cfg.width] -
                                    4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);
            next[idx] = 2.0f * current[idx] - previous[idx] + c2 * laplacian -
                        local_damping * velocity;
        }
    }
}

// Mismo calculo de render_frame del codigo base, pero devuelve solo el gris.
inline void cpu_render_gray(const std::vector<float>& h, const Config& cfg,
                            std::vector<unsigned char>& gray) {
    const float light_len = std::sqrt(0.35f * 0.35f + 0.55f * 0.55f + 0.76f * 0.76f);
    const float lx = -0.35f / light_len;
    const float ly = -0.55f / light_len;
    const float lz = 0.76f / light_len;

    gray.assign(static_cast<std::size_t>(cfg.width) * cfg.height, 0);
    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float nx = 2.8f * (h[index_of(xm, y, cfg.width)] - h[index_of(xp, y, cfg.width)]);
            const float ny = 2.8f * (h[index_of(x, ym, cfg.width)] - h[index_of(x, yp, cfg.width)]);
            const float inv_len = 1.0f / std::sqrt(nx * nx + ny * ny + 1.0f);

            const float diffuse = std::max(0.0f, nx * inv_len * lx + ny * inv_len * ly + inv_len * lz);
            const float wave = std::min(1.0f, std::max(0.0f, 0.5f + 1.8f * h[index_of(x, y, cfg.width)]));
            const float specular = std::pow(diffuse, 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;
            gray[index_of(x, y, cfg.width)] =
                static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, intensity)));
        }
    }
}

// Corre `steps` pasos en CPU y en GPU y compara la malla de alturas. El render
// se compara sobre la malla de la GPU, asi el error que sale es solo del kernel
// de render y no se mezcla con el de la simulacion.
//
// `Sim` tiene que ofrecer: upload(prev, cur), step(), download_heights(out) y
// render_gray(out).
template <class Sim>
void validate(Sim& sim, const Config& cfg, int steps) {
    const std::size_t cells = static_cast<std::size_t>(cfg.width) * cfg.height;
    std::vector<float> previous(cells, 0.0f);
    std::vector<float> current(cells, 0.0f);
    std::vector<float> next(cells, 0.0f);

    add_drop(current, previous, cfg);
    sim.upload(previous, current);

    for (int i = 0; i < steps; ++i) {
        cpu_step(previous, current, next, cfg);
        previous.swap(current);
        current.swap(next);
        sim.step();
    }

    std::vector<float> gpu_heights;
    sim.download_heights(gpu_heights);

    double max_err = 0.0;
    double sum_err = 0.0;
    for (std::size_t i = 0; i < cells; ++i) {
        const double err = std::fabs(static_cast<double>(gpu_heights[i]) - current[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    std::vector<unsigned char> gpu_gray;
    std::vector<unsigned char> cpu_gray;
    sim.render_gray(gpu_gray);
    cpu_render_gray(gpu_heights, cfg, cpu_gray);

    int max_pixel = 0;
    for (std::size_t i = 0; i < cells; ++i) {
        max_pixel = std::max(max_pixel, std::abs(static_cast<int>(gpu_gray[i]) - cpu_gray[i]));
    }

    std::cout << "Validacion contra CPU (" << steps << " pasos)\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Error maximo alturas: " << max_err << '\n';
    std::cout << "Error medio alturas: " << sum_err / static_cast<double>(cells) << '\n';
    std::cout << std::defaultfloat;
    std::cout << "Error maximo pixel: " << max_pixel << " (de 255)\n";
}
