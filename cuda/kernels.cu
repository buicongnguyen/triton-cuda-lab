// Standalone CUDA counterpart to the Triton labs. FP32, forward only.
// Reduction concepts: NVIDIA CUDA C++ Programming and Best Practices Guides.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) check((call), #call)
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Buffer {
    float* ptr = nullptr;
    explicit Buffer(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Event {
    cudaEvent_t value;
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

__global__ void vector_add(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

// Deliberately slow baseline: one CUDA thread serially handles a complete row.
__global__ void softmax_serial(const float* x, float* out, int rows, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    x += row * width;
    out += row * width;
    float maximum = -INFINITY;
    for (int j = 0; j < width; ++j) maximum = fmaxf(maximum, x[j]);
    // Compensated FP32 summation keeps long serial rows within the same numerical
    // contract as the parallel tree; max subtraction alone does not prevent drift.
    float total = 0.0f, correction = 0.0f;
    for (int j = 0; j < width; ++j) {
        float corrected = expf(x[j] - maximum) - correction;
        float next = total + corrected;
        correction = (next - total) - corrected;
        total = next;
    }
    for (int j = 0; j < width; ++j) out[j] = expf(x[j] - maximum) / total;
}

// The block reduction is written for this exact block size: every warp is full and
// warp 0 can combine one partial per warp. Change it here, never at a launch site.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
static_assert(kThreads % 32 == 0 && kWarps <= 32, "block_reduce needs full warps");

template<bool Maximum>
__device__ float combine(float a, float b) { return Maximum ? fmaxf(a, b) : a + b; }

template<bool Maximum>
__device__ float warp_reduce(float value) {
    // Every warp is full: softmax_parallel always launches exactly kThreads threads.
    for (int distance = 16; distance; distance /= 2)
        value = combine<Maximum>(value, __shfl_down_sync(0xffffffff, value, distance));
    return value;
}

template<bool Maximum>
__device__ float block_reduce(float value, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    value = warp_reduce<Maximum>(value);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < kWarps ? scratch[lane] : (Maximum ? -INFINITY : 0.0f);
        value = warp_reduce<Maximum>(value);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    float result = scratch[0];
    // All readers finish before a subsequent reduction reuses the scratch array.
    __syncthreads();
    return result;
}

__global__ void __launch_bounds__(kThreads) softmax_parallel(const float* x, float* out, int width) {
    __shared__ float scratch[kWarps];
    x += blockIdx.x * width;
    out += blockIdx.x * width;
    float maximum = -INFINITY;
    for (int j = threadIdx.x; j < width; j += blockDim.x) maximum = fmaxf(maximum, x[j]);
    maximum = block_reduce<true>(maximum, scratch);
    float total = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) total += expf(x[j] - maximum);
    total = block_reduce<false>(total, scratch);
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        out[j] = expf(x[j] - maximum) / total;
}

// Online statistics (the merge rule of workshop A1): one pass finds the row maximum and
// the sum of exponentials together, so the row is read twice instead of three times.
struct MaxSum {
    float max, sum;
};

__device__ MaxSum merge(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.max, b.max);
    if (m == -INFINITY) return {m, 0.0f};  // both empty: avoid (-inf) - (-inf)
    return {m, a.sum * expf(a.max - m) + b.sum * expf(b.max - m)};
}

// Masked (-inf) inputs leave the state unchanged until a finite value arrives, so a
// lane that starts inside a mask never computes (-inf) - (-inf).
__device__ MaxSum add_value(MaxSum s, float value) {
    const float m = fmaxf(s.max, value);
    if (m == -INFINITY) return s;
    return {m, s.sum * expf(s.max - m) + expf(value - m)};
}

__device__ MaxSum add_four(MaxSum s, float4 v) {
    const float m = fmaxf(s.max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    if (m == -INFINITY) return s;
    return {m, s.sum * expf(s.max - m) + expf(v.x - m) + expf(v.y - m) + expf(v.z - m)
                   + expf(v.w - m)};
}

__device__ MaxSum warp_merge(MaxSum v) {
    for (int distance = 16; distance; distance /= 2) {
        const MaxSum other{__shfl_down_sync(0xffffffff, v.max, distance),
                           __shfl_down_sync(0xffffffff, v.sum, distance)};
        v = merge(v, other);
    }
    return v;
}

__device__ MaxSum block_merge(MaxSum v, float* scratch_max, float* scratch_sum) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_merge(v);
    if (lane == 0) {
        scratch_max[warp] = v.max;
        scratch_sum[warp] = v.sum;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < kWarps ? MaxSum{scratch_max[lane], scratch_sum[lane]} : MaxSum{-INFINITY, 0.0f};
        v = warp_merge(v);
        if (lane == 0) {
            scratch_max[0] = v.max;
            scratch_sum[0] = v.sum;
        }
    }
    __syncthreads();
    // Called once per block, so the scratch is never reused and needs no trailing barrier.
    return {scratch_max[0], scratch_sum[0]};
}

// Vec4 reads four floats per 16-byte load; the launcher selects it when width % 4 == 0,
// which keeps every row start 16-byte aligned.
template<bool Vec4>
__global__ void __launch_bounds__(kThreads) softmax_online(const float* x, float* out, int width) {
    __shared__ float scratch_max[kWarps];
    __shared__ float scratch_sum[kWarps];
    x += blockIdx.x * width;
    out += blockIdx.x * width;
    MaxSum state{-INFINITY, 0.0f};
    if constexpr (Vec4) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = threadIdx.x; j < width / 4; j += kThreads) state = add_four(state, x4[j]);
    } else {
        for (int j = threadIdx.x; j < width; j += kThreads) state = add_value(state, x[j]);
    }
    const MaxSum row = block_merge(state, scratch_max, scratch_sum);
    const float inverse = 1.0f / row.sum;
    if constexpr (Vec4) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* out4 = reinterpret_cast<float4*>(out);
        for (int j = threadIdx.x; j < width / 4; j += kThreads) {
            const float4 v = x4[j];
            out4[j] = make_float4(expf(v.x - row.max) * inverse, expf(v.y - row.max) * inverse,
                                  expf(v.z - row.max) * inverse, expf(v.w - row.max) * inverse);
        }
    } else {
        for (int j = threadIdx.x; j < width; j += kThreads)
            out[j] = expf(x[j] - row.max) * inverse;
    }
}

void launch_softmax_online(const float* x, float* out, int rows, int width, bool allow_vec4 = true) {
    if (allow_vec4 && width % 4 == 0)
        softmax_online<true><<<rows, kThreads>>>(x, out, width);
    else
        softmax_online<false><<<rows, kThreads>>>(x, out, width);
}

std::vector<float> random_values(int count) {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
    std::vector<float> values(count);
    for (auto& value : values) value = distribution(rng);
    return values;
}

void upload(Buffer& dst, const std::vector<float>& src) {
    CUDA_CHECK(cudaMemcpy(dst.ptr, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
}

double verify_softmax(const char* kernel, const std::vector<float>& input, const Buffer& output,
                      int rows, int width) {
    std::vector<float> actual(input.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), output.ptr, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double maximum_error = 0.0;
    for (int i = 0; i < rows; ++i) {
        const float* x = input.data() + i * width;
        double maximum = *std::max_element(x, x + width), total = 0.0, probability_sum = 0.0;
        for (int j = 0; j < width; ++j) total += std::exp(double(x[j]) - maximum);
        for (int j = 0; j < width; ++j) {
            double expected = std::exp(double(x[j]) - maximum) / total;
            double value = actual[i * width + j];
            double error = std::abs(value - expected);
            if (!std::isfinite(value) || error > 2e-6 + 2e-5 * std::abs(expected))
                throw std::runtime_error(std::string(kernel) + " correctness failed: row " + std::to_string(i)
                                         + ", column " + std::to_string(j) + ", width " + std::to_string(width));
            maximum_error = std::max(maximum_error, error);
            probability_sum += value;
        }
        if (std::abs(probability_sum - 1.0) > 2e-5)
            throw std::runtime_error(std::string(kernel) + " row sum=" + std::to_string(probability_sum)
                                     + " width=" + std::to_string(width));
    }
    return maximum_error;
}

void test_add(int n) {
    auto x = random_values(n), y = random_values(n);
    // Make y distinct, including both signs.
    for (auto& value : y) value *= -0.5f;
    Buffer dx(n), dy(n), out(n);
    upload(dx, x);
    upload(dy, y);
    vector_add<<<(n + 255) / 256, 256>>>(dx.ptr, dy.ptr, out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> actual(n);
    CUDA_CHECK(cudaMemcpy(actual.data(), out.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(actual[i]) || actual[i] != x[i] + y[i])
            throw std::runtime_error("Vector add correctness failed");
}

// All-ones bytes are a NaN in every float, so an element a kernel fails to write
// cannot pass verification using an earlier kernel's result.
void poison(Buffer& buffer, size_t count) {
    CUDA_CHECK(cudaMemset(buffer.ptr, 0xFF, count * sizeof(float)));
}

void test_softmax(int rows, int width, bool extreme) {
    auto values = random_values(rows * width);
    if (extreme) {
        for (int j = 0; j < width; ++j) values[j] = (j % 2 ? 9999.0f : 10000.0f);
        if (rows > 1) std::fill(values.begin() + width, values.begin() + 2 * width, -9000.0f);
        // An attention-style mask: a masked prefix and scattered masked entries, never
        // the whole row (a fully masked row has no defined softmax).
        if (rows > 2)
            for (int j = 0; j < width; ++j)
                if (j < width / 3 || j % 5 == 1) values[2 * width + j] = -INFINITY;
    }
    Buffer x(values.size()), out(values.size());
    upload(x, values);
    poison(out, values.size());
    softmax_serial<<<(rows + 255) / 256, 256>>>(x.ptr, out.ptr, rows, width);
    CUDA_CHECK(cudaGetLastError());
    verify_softmax("softmax_serial", values, out, rows, width);
    poison(out, values.size());
    softmax_parallel<<<rows, kThreads>>>(x.ptr, out.ptr, width);
    CUDA_CHECK(cudaGetLastError());
    verify_softmax("softmax_parallel", values, out, rows, width);
    poison(out, values.size());
    launch_softmax_online(x.ptr, out.ptr, rows, width);
    CUDA_CHECK(cudaGetLastError());
    verify_softmax("softmax_online", values, out, rows, width);
}

constexpr int kSamples = 15, kLaunchesPerSample = 50;

// Times all kernels together: each round takes one sample of every kernel, starting at
// a different kernel each time. On a GPU shared with desktop applications, a burst of
// other work then costs one sample of several kernels, which the median discards,
// instead of every sample of one kernel.
std::vector<std::vector<float>> benchmark(const std::vector<std::function<void()>>& launches) {
    for (const auto& launch : launches)
        for (int i = 0; i < 10; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Event start, end;
    std::vector<std::vector<float>> samples(launches.size());
    for (int sample = 0; sample < kSamples; ++sample) {
        for (size_t k = 0; k < launches.size(); ++k) {
            const size_t v = (k + sample) % launches.size();
            CUDA_CHECK(cudaEventRecord(start.value));
            for (int i = 0; i < kLaunchesPerSample; ++i) launches[v]();
            CUDA_CHECK(cudaEventRecord(end.value));
            CUDA_CHECK(cudaEventSynchronize(end.value));
            CUDA_CHECK(cudaGetLastError());
            float elapsed;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.value, end.value));
            samples[v].push_back(elapsed / kLaunchesPerSample);
        }
    }
    return samples;
}

void write_result(std::ostream& stream, const char* name, std::vector<float> samples,
                  bool comma) {
    stream << (comma ? ",\n" : "\n") << "    \"" << name << "\": {\"samples_ms\": [";
    for (size_t i = 0; i < samples.size(); ++i) stream << (i ? ", " : "") << samples[i];
    std::sort(samples.begin(), samples.end());
    stream << "], \"median_ms\": " << samples[samples.size() / 2] << "}";
}

int main(int argc, char** argv) {
    try {
        bool test_only = argc == 2 && std::string(argv[1]) == "--test-only";
        bool file_output = argc == 3 && std::string(argv[1]) == "--json";
        if (argc != 1 && !test_only && !file_output)
            throw std::runtime_error("Usage: cuda_portfolio [--test-only | --json path]");
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        for (int n : {1, 257, 65537}) test_add(n);
        for (int width : {1, 33, 127, 1024, 4097, 8192}) test_softmax(7, width, true);
        if (test_only) { std::cout << "CUDA correctness passed (3 vector + 18 softmax cases)\n"; return 0; }
        const int rows = 1024, width = 1024, count = rows * width;
        test_add(count);
        test_softmax(rows, width, false);
        auto values = random_values(count);
        Buffer x(count), y(count), out(count);
        upload(x, values);
        upload(y, values);
        const auto times = benchmark({
            [&] { vector_add<<<(count + 255) / 256, 256>>>(x.ptr, y.ptr, out.ptr, count); },
            [&] { softmax_serial<<<(rows + 255) / 256, 256>>>(x.ptr, out.ptr, rows, width); },
            [&] { softmax_parallel<<<rows, kThreads>>>(x.ptr, out.ptr, width); },
            [&] { launch_softmax_online(x.ptr, out.ptr, rows, width); },
            [&] { launch_softmax_online(x.ptr, out.ptr, rows, width, false); },
        });
        int driver, runtime;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ofstream file;
        if (file_output) { file.open(argv[2]); if (!file) throw std::runtime_error("Cannot open JSON output"); }
        std::ostream& stream = file_output ? file : std::cout;
        stream << std::setprecision(9) << "{\n  \"gpu\": \"" << props.name
               << "\",\n  \"driver\": " << driver << ",\n  \"runtime\": " << runtime
               << ",\n  \"dtype\": \"float32\",\n  \"timing\": \"CUDA events; 50 launches/sample; 15 samples, kernels interleaved\","
               << "\n  \"softmax_shape\": [1024, 1024],\n  \"add_elements\": 1048576,"
               << "\n  \"correctness\": \"passed CPU double softmax reference and exact float addition\","
               << "\n  \"variants\": {";
        write_result(stream, "vector_add", times[0], false);
        write_result(stream, "softmax_serial", times[1], true);
        write_result(stream, "softmax_parallel", times[2], true);
        write_result(stream, "softmax_online", times[3], true);
        write_result(stream, "softmax_online_scalar", times[4], true);
        stream << "\n  }\n}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
