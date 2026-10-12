// Version 4: la GPU trabaja mientras la CPU codifica el video.
// Los kernels son los de v3_aritmetica.cu. Lo que cambia es el ciclo principal:
//   - todo el trabajo de GPU va en un stream propio y la copia D2H es
//     asincrona (cudaMemcpyAsync) hacia memoria pinned;
//   - hay dos buffers pinned: mientras la CPU codifica el cuadro N, la GPU ya
//     esta simulando, renderizando y copiando el cuadro N+1.
// El resultado es el mismo video; solo cambia el orden en que se hace el trabajo.
#include "common.h"

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kTileW = kBlockX + 2;
constexpr int kTileH = kBlockY + 2;

// Copia a `tile` la parte de `src` que le toca al bloque, con una celda extra
// alrededor. Fuera de la malla se repite el valor del borde (igual que el
// render original), y los hilos del bloque se reparten la carga de forma que
// los accesos consecutivos caen en direcciones consecutivas.
__device__ void load_tile(const float* __restrict__ src, float (*tile)[kTileW],
                          int width, int height, int x0, int y0) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    for (int i = tid; i < kTileW * kTileH; i += threads) {
        const int ty = i / kTileW;
        const int tx = i % kTileW;
        const int gx = min(max(x0 - 1 + tx, 0), width - 1);
        const int gy = min(max(y0 - 1 + ty, 0), height - 1);
        tile[ty][tx] = src[gy * width + gx];
    }
}

__global__ void simulate_step_kernel(const float* __restrict__ previous,
                                     const float* __restrict__ current,
                                     float* __restrict__ next, int width, int height,
                                     float wave_speed, float damping,
                                     float edge_damping) {
    __shared__ float tile[kTileH][kTileW];

    const int x0 = blockIdx.x * kBlockX;
    const int y0 = blockIdx.y * kBlockY;
    load_tile(current, tile, width, height, x0, y0);
    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;

    // Los bordes se mantienen en cero.
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        next[idx] = 0.0f;
        return;
    }

    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const float center = tile[ty][tx];

    const float c2 = wave_speed * wave_speed;

    const float laplacian = tile[ty][tx - 1] + tile[ty][tx + 1] +
                            tile[ty - 1][tx] + tile[ty + 1][tx] - 4.0f * center;

    const float velocity = center - previous[idx];

    const int dist = min(min(x, y), min(width - 1 - x, height - 1 - y));
    float local_damping = damping;
    if (dist < 32) {
        const float t = 1.0f - static_cast<float>(dist) / 32.0f;
        local_damping += edge_damping * t * t;
    }

    next[idx] = 2.0f * center - previous[idx] + c2 * laplacian - local_damping * velocity;
}

// x^24 = (x^8)^3 = x^16 * x^8
__device__ float pow24(float x) {
    const float x2 = x * x;
    const float x4 = x2 * x2;
    const float x8 = x4 * x4;
    const float x16 = x8 * x8;
    return x16 * x8;
}

// Gris (0-255) de un pixel a partir de la pendiente de la superficie (dx, dy)
// y de su altura h.
__device__ unsigned char shade_pixel(float dx, float dy, float h) {
    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;

    const float inv_normal_length = rsqrtf(nx * nx + ny * ny + nz * nz);
    const float nnx = nx * inv_normal_length;
    const float nny = ny * inv_normal_length;
    const float nnz = nz * inv_normal_length;

    const float light_length = sqrtf(0.35f * 0.35f + 0.55f * 0.55f + 0.76f * 0.76f);
    const float lx = -0.35f / light_length;
    const float ly = -0.55f / light_length;
    const float lz = 0.76f / light_length;

    float diffuse = nnx * lx + nny * ly + nnz * lz;
    diffuse = fmaxf(0.0f, diffuse);

    float wave = 0.5f + 1.8f * h;
    wave = fminf(1.0f, fmaxf(0.0f, wave));

    const float specular = pow24(diffuse);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;
    intensity = fminf(255.0f, fmaxf(0.0f, intensity));

    return static_cast<unsigned char>(intensity);
}

__global__ void render_frame_kernel(const float* __restrict__ height,
                                    unsigned char* __restrict__ image, int width,
                                    int height_size) {
    __shared__ float tile[kTileH][kTileW];

    const int x0 = blockIdx.x * kBlockX;
    const int y0 = blockIdx.y * kBlockY;
    load_tile(height, tile, width, height_size, x0, y0);
    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;

    if (x >= width || y >= height_size) {
        return;
    }

    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;

    const float dx = tile[ty][tx - 1] - tile[ty][tx + 1];
    const float dy = tile[ty - 1][tx] - tile[ty + 1][tx];

    image[y * width + x] = shade_pixel(dx, dy, tile[ty][tx]);
}

class DeviceSim {
public:
    explicit DeviceSim(const Config& cfg)
        : cfg_(cfg),
          cells_(static_cast<std::size_t>(cfg.width) * cfg.height),
          block_(kBlockX, kBlockY),
          grid_((cfg.width + kBlockX - 1) / kBlockX, (cfg.height + kBlockY - 1) / kBlockY) {
        cuda_check(cudaMalloc(&previous_, grid_bytes()), "cudaMalloc previous");
        cuda_check(cudaMalloc(&current_, grid_bytes()), "cudaMalloc current");
        cuda_check(cudaMalloc(&next_, grid_bytes()), "cudaMalloc next");
        cuda_check(cudaMalloc(&image_, cells_), "cudaMalloc image");
        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~DeviceSim() {
        cudaStreamDestroy(stream_);
        cudaFree(previous_);
        cudaFree(current_);
        cudaFree(next_);
        cudaFree(image_);
    }

    DeviceSim(const DeviceSim&) = delete;
    DeviceSim& operator=(const DeviceSim&) = delete;

    void upload(const std::vector<float>& previous, const std::vector<float>& current) {
        cuda_check(cudaMemcpy(previous_, previous.data(), grid_bytes(), cudaMemcpyHostToDevice),
                   "cudaMemcpy previous");
        cuda_check(cudaMemcpy(current_, current.data(), grid_bytes(), cudaMemcpyHostToDevice),
                   "cudaMemcpy current");
        cuda_check(cudaMemset(next_, 0, grid_bytes()), "cudaMemset next");
    }

    cudaStream_t stream() const { return stream_; }

    // Los punteros se rotan al encolar el paso, no cuando termina en la GPU;
    // cada kernel ya recibio los suyos como argumentos, asi que no hay problema.
    void step() {
        simulate_step_kernel<<<grid_, block_, 0, stream_>>>(previous_, current_, next_,
                                                            cfg_.width, cfg_.height,
                                                            cfg_.wave_speed, cfg_.damping,
                                                            cfg_.edge_damping);
        cuda_check(cudaGetLastError(), "simulate_step_kernel");

        float* tmp = previous_;
        previous_ = current_;
        current_ = next_;
        next_ = tmp;
    }

    void render() {
        render_frame_kernel<<<grid_, block_, 0, stream_>>>(current_, image_, cfg_.width,
                                                           cfg_.height);
        cuda_check(cudaGetLastError(), "render_frame_kernel");
    }

    // `host` debe ser memoria pinned, si no la copia deja de ser asincrona.
    void copy_image_async(unsigned char* host) {
        cuda_check(cudaMemcpyAsync(host, image_, cells_, cudaMemcpyDeviceToHost, stream_),
                   "cudaMemcpyAsync image");
    }

    // Version bloqueante, solo para validar.
    void copy_image(unsigned char* host) {
        cuda_check(cudaMemcpy(host, image_, cells_, cudaMemcpyDeviceToHost), "cudaMemcpy image");
    }

    void download_heights(std::vector<float>& out) {
        out.resize(cells_);
        cuda_check(cudaMemcpy(out.data(), current_, grid_bytes(), cudaMemcpyDeviceToHost),
                   "cudaMemcpy heights");
    }

    void render_gray(std::vector<unsigned char>& out) {
        out.resize(cells_);
        render();
        copy_image(out.data());
    }

private:
    std::size_t grid_bytes() const { return cells_ * sizeof(float); }

    const Config& cfg_;
    std::size_t cells_;
    dim3 block_;
    dim3 grid_;
    float* previous_ = nullptr;
    float* current_ = nullptr;
    float* next_ = nullptr;
    unsigned char* image_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

// Buffer del host en memoria pinned.
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) {
        cuda_check(cudaMallocHost(&ptr_, bytes), "cudaMallocHost");
    }
    ~PinnedBuffer() { cudaFreeHost(ptr_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    unsigned char* data() { return ptr_; }

private:
    unsigned char* ptr_ = nullptr;
};

// Todo lo que necesita un cuadro en vuelo: su buffer del host y sus tiempos.
struct Slot {
    explicit Slot(std::size_t bytes) : host(bytes) {}
    PinnedBuffer host;
    EventTimer sim, render, copy;
};

int main(int argc, char** argv) {
    try {
        Config cfg;
        cfg.output = "output/v4_async.mp4";
        const Options opt = parse_args(argc, argv, cfg);

        const int frames = total_frames(cfg);
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * cfg.height;

        DeviceSim sim(cfg);

        if (opt.validate) {
            validate(sim, cfg, frames * cfg.steps_per_frame);
            return 0;
        }

        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        add_drop(current, previous, cfg);
        sim.upload(previous, current);

        cv::VideoWriter writer;
        open_writer(writer, cfg);

        cv::Mat bgr(cfg.height, cfg.width, CV_8UC3);
        Slot slot_a(cells);
        Slot slot_b(cells);
        Slot* slots[2] = {&slot_a, &slot_b};
        const cudaStream_t stream = sim.stream();
        Totals total;

        // Deja en cola todo el trabajo de GPU de un cuadro y regresa sin esperar.
        auto enqueue_frame = [&](int frame) {
            Slot& s = *slots[frame % 2];

            s.sim.begin(stream);
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                sim.step();
            }
            s.sim.end(stream);

            s.render.begin(stream);
            sim.render();
            s.render.end(stream);

            s.copy.begin(stream);
            sim.copy_image_async(s.host.data());
            s.copy.end(stream);
        };

        const auto start = Clock::now();

        if (frames > 0) {
            enqueue_frame(0);
        }

        for (int frame = 0; frame < frames; ++frame) {
            // La GPU arranca el siguiente cuadro antes de que la CPU codifique este.
            if (frame + 1 < frames) {
                enqueue_frame(frame + 1);
            }

            Slot& s = *slots[frame % 2];

            const auto t_wait = Clock::now();
            s.copy.wait();
            total.wait += since(t_wait);

            cv::Mat image(cfg.height, cfg.width, CV_8UC1, s.host.data());
            const auto t0 = Clock::now();
            put_label(image, frame);
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
            writer.write(bgr);
            total.encode += since(t0);

            total.sim += s.sim.seconds();
            total.render += s.render.seconds();
            total.copy += s.copy.seconds();

            if (frame % std::max(1, frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << frames << '\n';
            }
        }

        writer.release();
        print_summary(cfg, since(start), total);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
