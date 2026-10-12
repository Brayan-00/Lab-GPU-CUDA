#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cuda_runtime.h>

namespace {

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

void create_parent_directories(const std::string& file_path) {
    const std::size_t last_separator = file_path.find_last_of('/');
    if (last_separator == std::string::npos) {
        return;
    }

    const std::string directory =
        last_separator == 0 ? "/" : file_path.substr(0, last_separator);
    std::string current;
    std::size_t position = 0;

    if (!directory.empty() && directory[0] == '/') {
        current = "/";
        position = 1;
    }

    while (position < directory.size()) {
        const std::size_t next_separator = directory.find('/', position);
        const std::size_t end =
            next_separator == std::string::npos ? directory.size() : next_separator;
        const std::string component = directory.substr(position, end - position);

        if (!component.empty()) {
            if (!current.empty() && current[current.size() - 1] != '/') {
                current += '/';
            }
            current += component;

            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error(
                    "No se pudo crear el directorio '" + current + "': " +
                    std::strerror(errno)
                );
            }

            struct stat info;
            if (stat(current.c_str(), &info) != 0) {
                throw std::runtime_error(
                    "No se pudo comprobar el directorio '" + current + "': " +
                    std::strerror(errno)
                );
            }
            if (!S_ISDIR(info.st_mode)) {
                throw std::runtime_error(
                    "La ruta existe, pero no es un directorio: " + current
                );
            }
        }

        if (next_separator == std::string::npos) {
            break;
        }
        position = next_separator + 1;
    }
}

void save_grid_to_binary(
    const std::string& filename,
    const float* grid,
    std::size_t size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "No se pudo abrir el archivo binario: " + filename
        );
    }

    file.write(
        reinterpret_cast<const char*>(grid),
        static_cast<std::streamsize>(size * sizeof(float))
    );
    if (!file) {
        throw std::runtime_error(
            "No se pudo escribir la malla en: " + filename
        );
    }

    file.close();
    if (!file) {
        throw std::runtime_error(
            "No se pudo cerrar el archivo binario: " + filename
        );
    }
}

int index_of(int x, int y, int width) {
    return y * width + x;
}

void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
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

float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

__device__ float border_absorption_device(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = min(min(x, y), min(cfg.width - 1 - x, cfg.height - 1 - y));
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

// Esta funcion actualiza la ecuacion de onda
void simulate_step(const std::vector<float>& previous,
                   const std::vector<float>& current,
                   std::vector<float>& next,
                   const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;

    std::fill(next.begin(), next.end(), 0.0f);


    // Lo más importante es el cálculo de next que depende de current y previous, que implican accesos a memoria
    // y las operaciones laplacian, que es paralelizable.
    // velocity que es paralelizable pero requiere menos cálculos.
    // local_damping que es paralelizable pero es una función externa.
    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian =
                current[idx - 1] + current[idx + 1] +
                current[idx - cfg.width] + current[idx + cfg.width] -
                4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);

            next[idx] = 2.0f * current[idx] - previous[idx] +
                        c2 * laplacian -
                        local_damping * velocity;
        }
    }
}


__global__ void simulate_step_kernel(const float* previous, const float* current,
    float* next, int width,
    int height, float wave_speed,
    float damping, float edge_damping)
{
    // Coordenadas de la celda que procesa este hilo.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evitar accesos fuera de los límites de la matriz.
    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;

    // Mantener los bordes en cero.
    if (x == 0 || x == width - 1 ||
        y == 0 || y == height - 1) {
        next[idx] = 0.0f;
        return;
    }

    // Cuadrado de la velocidad de propagación.
    const float c2 = wave_speed * wave_speed;

    // Laplaciano espacial: vecinos izquierdo, derecho,
    // superior e inferior.
    const float laplacian =
        current[idx - 1] +
        current[idx + 1] +
        current[idx - width] +
        current[idx + width] -
        4.0f * current[idx];

    // Diferencia temporal entre los dos estados anteriores.
    const float velocity = current[idx] - previous[idx];

    // Distancia mínima al borde de la matriz.
    const int dist = min(
        min(x, y),
        min(width - 1 - x, height - 1 - y)
    );

    // Amortiguamiento local.
    float local_damping = damping;

    if (dist < 32) {
        const float t =
            1.0f - static_cast<float>(dist) / 32.0f;

        local_damping += edge_damping * t * t;
    }

    // Ecuación de actualización de la onda.
    next[idx] =
        2.0f * current[idx] -
        previous[idx] +
        c2 * laplacian -
        local_damping * velocity;
}


// Esta funcion se encarga de renderizar la altura de la superficie en una imagen en escala de grises
cv::Mat render_frame(const std::vector<float>& height, const Config& cfg, int frame_number) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float dx = height[index_of(xm, y, cfg.width)] - height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] - height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal = cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f));

            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            const float wave = std::max(
                0.0f,
                std::min(
                    1.0f,
                    0.5f + 1.8f * height[index_of(x, y, cfg.width)]
                )
            );
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, intensity))
            );
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }

    cv::putText(image,
                "CPU float32 | frame " + std::to_string(frame_number),
                cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA);

    return image;
}

}  // namespace


__global__ void render_frame_kernel(
    const float* height,
    unsigned char* image,
    int width,
    int height_size)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height_size) {
        return;
    }

    const int idx = y * width + x;

    const int xm = max(0, x - 1);
    const int xp = min(width - 1, x + 1);
    const int ym = max(0, y - 1);
    const int yp = min(height_size - 1, y + 1);

    const float dx =
        height[y * width + xm] - height[y * width + xp];

    const float dy =
        height[ym * width + x] - height[yp * width + x];

    // Normal de la superficie.
    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;

    const float inv_normal_length =
        rsqrtf(nx * nx + ny * ny + nz * nz);

    const float nnx = nx * inv_normal_length;
    const float nny = ny * inv_normal_length;
    const float nnz = nz * inv_normal_length;

    // Dirección de luz normalizada, igual que en la función original.
    const float light_length =
        sqrtf(0.35f * 0.35f + 0.55f * 0.55f + 0.76f * 0.76f);

    const float lx = -0.35f / light_length;
    const float ly = -0.55f / light_length;
    const float lz =  0.76f / light_length;

    float diffuse = nnx * lx + nny * ly + nnz * lz;
    diffuse = fmaxf(0.0f, diffuse);

    float wave = 0.5f + 1.8f * height[idx];
    wave = fminf(1.0f, fmaxf(0.0f, wave));

    const float specular = powf(diffuse, 24.0f);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;

    intensity = fminf(255.0f, fmaxf(0.0f, intensity));

    const unsigned char gray =
        static_cast<unsigned char>(intensity);

    // OpenCV CV_8UC3: tres bytes consecutivos por píxel (B, G, R).
    const int pixel_idx = 3 * idx;
    image[pixel_idx]     = gray;
    image[pixel_idx + 1] = gray;
    image[pixel_idx + 2] = gray;
}


cv::Mat render_frame_cuda(
    const float* d_height,
    const Config& cfg,
    int frame_number,
    unsigned char* d_image)
{
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks(
        (cfg.width + threads_per_block.x - 1) /
            threads_per_block.x,
        (cfg.height + threads_per_block.y - 1) /
            threads_per_block.y
    );

    render_frame_kernel<<<num_blocks, threads_per_block>>>(
        d_height,
        d_image,
        cfg.width,
        cfg.height
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Error al lanzar render_frame_kernel: ") +
            cudaGetErrorString(err)
        );
    }

    err = cudaMemcpy(
        image.data,
        d_image,
        image.total() * image.elemSize(),
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Error al copiar la imagen desde GPU: ") +
            cudaGetErrorString(err)
        );
    }

    // El texto se dibuja en CPU, después de transferir la imagen.
    cv::putText(
        image,
        "CUDA | frame " + std::to_string(frame_number),
        cv::Point(18, 32),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(235, 235, 235),
        1,
        cv::LINE_AA
    );

    return image;
}

int main() {
    float* d_previous = nullptr;
    float* d_current = nullptr;
    float* d_next = nullptr;
    unsigned char* d_render_image = nullptr;

    auto check_cuda = [](cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    };

    try {
        const Config cfg;

        const int total_frames =
            static_cast<int>(std::round(cfg.seconds * cfg.fps));

        const std::size_t cells =
            static_cast<std::size_t>(cfg.width) *
            static_cast<std::size_t>(cfg.height);

        const std::size_t bytes = cells * sizeof(float);
        const std::size_t image_bytes =
            cells * 3 * sizeof(unsigned char);

        const dim3 threadsPerBlock(16, 16);

        const dim3 numBlocks(
            (cfg.width + threadsPerBlock.x - 1) /
                threadsPerBlock.x,
            (cfg.height + threadsPerBlock.y - 1) /
                threadsPerBlock.y
        );

        create_parent_directories(cfg.output);

        // Buffers de CPU.
        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        std::vector<float> next(cells, 0.0f);

        // Inicializar la perturbación en CPU.
        add_drop(current, previous, cfg);

        // Reservar los buffers de GPU.
        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_previous), bytes));

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_current), bytes));

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_next), bytes));

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_render_image), image_bytes));

        // Copiar las condiciones iniciales al dispositivo.
        check_cuda(cudaMemcpy(
            d_previous,
            previous.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        check_cuda(cudaMemcpy(
            d_current,
            current.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        check_cuda(cudaMemset(d_next, 0, bytes));

        // Preparar el archivo de video.
        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height)
        );

        if (!writer.isOpened()) {
            throw std::runtime_error(
                "No se pudo abrir el archivo de salida: " +
                cfg.output
            );
        }

        const auto start = std::chrono::steady_clock::now();

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0;
                 step < cfg.steps_per_frame;
                 ++step) {

                simulate_step_kernel<<<numBlocks, threadsPerBlock>>>(
                    d_previous,
                    d_current,
                    d_next,
                    cfg.width,
                    cfg.height,
                    cfg.wave_speed,
                    cfg.damping,
                    cfg.edge_damping
                );

                check_cuda(cudaGetLastError());
                check_cuda(cudaDeviceSynchronize());

                // Intercambiar los buffers de la simulación.
                float* temp = d_previous;
                d_previous = d_current;
                d_current = d_next;
                d_next = temp;
            }

            // Renderizar la imagen utilizando el kernel CUDA.
            cv::Mat rendered_frame = render_frame_cuda(
                d_current,
                cfg,
                frame,
                d_render_image
            );

            // Codificación del video en CPU.
            writer.write(rendered_frame);

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame
                          << " / " << total_frames << '\n';
            }
        }

        // Finalizar la escritura antes de informar el resultado.
        writer.release();

        const auto end = std::chrono::steady_clock::now();

        const double elapsed =
            std::chrono::duration<double>(end - start).count();

        std::vector<float> final_grid(cells);
        check_cuda(cudaMemcpy(
            final_grid.data(),
            d_current,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const std::string grid_output = "output/final_grid_gpu.bin";
        save_grid_to_binary(grid_output, final_grid.data(), cells);

        const double simulated_steps =
            static_cast<double>(total_frames) *
            cfg.steps_per_frame;

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Malla final guardada: " << grid_output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: "
                  << simulated_steps << '\n';
        std::cout << "Rendimiento: "
                  << simulated_steps / elapsed
                  << " pasos/s\n";

        // Liberar los buffers CUDA.
        check_cuda(cudaFree(d_previous));
        d_previous = nullptr;

        check_cuda(cudaFree(d_current));
        d_current = nullptr;

        check_cuda(cudaFree(d_next));
        d_next = nullptr;

        check_cuda(cudaFree(d_render_image));
        d_render_image = nullptr;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';

        // Limpieza segura ante errores.
        if (d_previous != nullptr) {
            cudaFree(d_previous);
        }

        if (d_current != nullptr) {
            cudaFree(d_current);
        }

        if (d_next != nullptr) {
            cudaFree(d_next);
        }

        if (d_render_image != nullptr) {
            cudaFree(d_render_image);
        }

        return 1;
    }

    return 0;
}
