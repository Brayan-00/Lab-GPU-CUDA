// Version 1: menos datos entre GPU y CPU.
// Cambios respecto a v0_base.cu:
//   - el render escribe 1 byte por pixel (gris) en vez de 3, asi la copia
//     D2H baja de 1.2 MB a 0.4 MB por cuadro;
//   - se quita el cudaDeviceSynchronize de cada paso (los kernels del mismo
//     stream ya se ejecutan en orden y cudaMemcpy espera por ellos);
//   - los cv::Mat del host se reutilizan y la CPU pasa el gris a BGR justo
//     antes de escribir al video (el VideoWriter trabaja en color).
#include "common.h"

__global__ void simulate_step_kernel(const float* previous, const float* current,
                                     float* next, int width, int height,
                                     float wave_speed, float damping,
                                     float edge_damping) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int idx = y * width + x;

    // Los bordes se mantienen en cero.
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        next[idx] = 0.0f;
        return;
    }

    const float c2 = wave_speed * wave_speed;

    const float laplacian = current[idx - 1] + current[idx + 1] +
                            current[idx - width] + current[idx + width] -
                            4.0f * current[idx];

    const float velocity = current[idx] - previous[idx];

    const int dist = min(min(x, y), min(width - 1 - x, height - 1 - y));
    float local_damping = damping;
    if (dist < 32) {
        const float t = 1.0f - static_cast<float>(dist) / 32.0f;
        local_damping += edge_damping * t * t;
    }

    next[idx] = 2.0f * current[idx] - previous[idx] + c2 * laplacian -
                local_damping * velocity;
}

// Igual que en v0, pero una sola escritura de 1 byte por pixel.
__global__ void render_frame_kernel(const float* height, unsigned char* image,
                                    int width, int height_size) {
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

    const float dx = height[y * width + xm] - height[y * width + xp];
    const float dy = height[ym * width + x] - height[yp * width + x];

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

    float wave = 0.5f + 1.8f * height[idx];
    wave = fminf(1.0f, fmaxf(0.0f, wave));

    const float specular = powf(diffuse, 24.0f);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;
    intensity = fminf(255.0f, fmaxf(0.0f, intensity));

    image[idx] = static_cast<unsigned char>(intensity);
}

class DeviceSim {
public:
    explicit DeviceSim(const Config& cfg)
        : cfg_(cfg),
          cells_(static_cast<std::size_t>(cfg.width) * cfg.height),
          block_(16, 16),
          grid_((cfg.width + 15) / 16, (cfg.height + 15) / 16) {
        cuda_check(cudaMalloc(&previous_, grid_bytes()), "cudaMalloc previous");
        cuda_check(cudaMalloc(&current_, grid_bytes()), "cudaMalloc current");
        cuda_check(cudaMalloc(&next_, grid_bytes()), "cudaMalloc next");
        cuda_check(cudaMalloc(&image_, cells_), "cudaMalloc image");
    }

    ~DeviceSim() {
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

    void step() {
        simulate_step_kernel<<<grid_, block_>>>(previous_, current_, next_, cfg_.width,
                                                cfg_.height, cfg_.wave_speed, cfg_.damping,
                                                cfg_.edge_damping);
        cuda_check(cudaGetLastError(), "simulate_step_kernel");

        float* tmp = previous_;
        previous_ = current_;
        current_ = next_;
        next_ = tmp;
    }

    void render() {
        render_frame_kernel<<<grid_, block_>>>(current_, image_, cfg_.width, cfg_.height);
        cuda_check(cudaGetLastError(), "render_frame_kernel");
    }

    // Copia la imagen en gris (1 byte por pixel) al host.
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
};

int main(int argc, char** argv) {
    try {
        Config cfg;
        cfg.output = "output/v1_menos_datos.mp4";
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

        cv::Mat image(cfg.height, cfg.width, CV_8UC1);
        cv::Mat bgr(cfg.height, cfg.width, CV_8UC3);
        EventTimer t_sim, t_render, t_copy;
        Totals total;

        const auto start = Clock::now();

        for (int frame = 0; frame < frames; ++frame) {
            t_sim.begin();
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                sim.step();
            }
            t_sim.end();

            t_render.begin();
            sim.render();
            t_render.end();

            t_copy.begin();
            sim.copy_image(image.data);
            t_copy.end();

            const auto t0 = Clock::now();
            put_label(image, frame);
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
            writer.write(bgr);
            total.encode += since(t0);

            total.sim += t_sim.seconds();
            total.render += t_render.seconds();
            total.copy += t_copy.seconds();

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
