#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#if defined(_WIN32) && !defined(_stdcall)
#define _stdcall __stdcall
#endif

#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif

#ifndef _Outptr_result_buffer_maybenull_
#define _Outptr_result_buffer_maybenull_(size)
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "onnxruntime_c_api.h"

namespace {

constexpr int kInputWidth = 224;
constexpr int kInputHeight = 224;
constexpr int kInputChannels = 3;
constexpr int64_t kElementCount =
    static_cast<int64_t>(kInputChannels) * kInputHeight * kInputWidth;

constexpr int kDogIndexStart = 151;
constexpr int kDogIndexEnd = 268;
constexpr int kCatIndexStart = 281;
constexpr int kCatIndexEnd = 285;
constexpr double kStrongOtherProbability = 0.90;

const char* kInputNames[] = {"inputs"};
const char* kOutputNames[] = {"output_0"};
const wchar_t* kModelPath = L"E:\\code\\c++\\mobilenetv2.onnx";

using OrtGetApiBaseFn = const OrtApiBase* (ORT_API_CALL*)();

bool check_status(const OrtApi* api, OrtStatus* status, const std::string& action) {
    if (status == nullptr) {
        return true;
    }

    std::cerr << action << " failed: " << api->GetErrorMessage(status) << std::endl;
    api->ReleaseStatus(status);
    return false;
}

float lanczos_kernel(float value) {
    constexpr float kPi = 3.14159265358979323846f;
    value = std::fabs(value);

    if (value < 1e-6f) {
        return 1.0f;
    }
    if (value >= 3.0f) {
        return 0.0f;
    }

    const float pi_value = kPi * value;
    return 3.0f * std::sin(pi_value) * std::sin(pi_value / 3.0f) /
           (pi_value * pi_value);
}

using WeightList = std::vector<std::pair<int, double>>;

std::vector<WeightList> build_lanczos_weights(
    int source_size,
    int target_size) {
    std::vector<WeightList> weights(static_cast<size_t>(target_size));
    const double scale = static_cast<double>(source_size) / target_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = 3.0 * filter_scale;

    for (int target_index = 0; target_index < target_size; ++target_index) {
        const double center =
            static_cast<double>(target_index) * scale +
            0.5 * (scale - 1.0);
        const int left = static_cast<int>(
            std::floor(center - support + 0.5));
        const int right = static_cast<int>(
            std::ceil(center + support - 0.5));
        double total_weight = 0.0;

        for (int source_index = left; source_index <= right; ++source_index) {
            if (source_index < 0 || source_index >= source_size) {
                continue;
            }

            const double distance =
                (static_cast<double>(source_index) - center) / filter_scale;
            const double weight = lanczos_kernel(
                static_cast<float>(distance));
            if (std::fabs(weight) < 1e-12) {
                continue;
            }

            weights[static_cast<size_t>(target_index)].emplace_back(
                source_index, weight);
            total_weight += weight;
        }

        if (total_weight == 0.0) {
            weights[static_cast<size_t>(target_index)].emplace_back(
                std::max(0, std::min(source_size - 1,
                                     static_cast<int>(center + 0.5))),
                1.0);
        } else {
            for (auto& item : weights[static_cast<size_t>(target_index)]) {
                item.second /= total_weight;
            }
        }
    }

    return weights;
}

unsigned char clamp_to_uint8(double value) {
    value = std::max(0.0, std::min(255.0, std::floor(value + 0.5)));
    return static_cast<unsigned char>(value);
}

std::vector<float> preprocess_image(
    const unsigned char* image,
    int width,
    int height) {
    const std::vector<WeightList> x_weights =
        build_lanczos_weights(width, kInputWidth);
    const std::vector<WeightList> y_weights =
        build_lanczos_weights(height, kInputHeight);

    std::vector<unsigned char> horizontal(
        static_cast<size_t>(height) * kInputWidth * kInputChannels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < kInputWidth; ++x) {
            for (int channel = 0; channel < kInputChannels; ++channel) {
                double value = 0.0;
                for (const auto& contribution : x_weights[x]) {
                    const int source_x = contribution.first;
                    const size_t source_index =
                        (static_cast<size_t>(y) * width + source_x) *
                            kInputChannels +
                        channel;
                    value += static_cast<double>(image[source_index]) *
                             contribution.second;
                }

                const size_t target_index =
                    (static_cast<size_t>(y) * kInputWidth + x) *
                        kInputChannels +
                    channel;
                horizontal[target_index] = clamp_to_uint8(value);
            }
        }
    }

    std::vector<float> tensor(static_cast<size_t>(kElementCount));

    for (int y = 0; y < kInputHeight; ++y) {
        for (int x = 0; x < kInputWidth; ++x) {
            for (int channel = 0; channel < kInputChannels; ++channel) {
                double value = 0.0;
                for (const auto& contribution : y_weights[y]) {
                    const int source_y = contribution.first;
                    const size_t source_index =
                        (static_cast<size_t>(source_y) * kInputWidth + x) *
                            kInputChannels +
                        channel;
                    value += static_cast<double>(horizontal[source_index]) *
                             contribution.second;
                }

                const unsigned char pixel = clamp_to_uint8(value);
                const size_t target_index =
                    (static_cast<size_t>(y) * kInputWidth + x) *
                        kInputChannels +
                    channel;
                tensor[target_index] =
                    static_cast<float>(pixel) / 127.5f - 1.0f;
            }
        }
    }

    return tensor;
}

void softmax_in_place(std::vector<float>& values) {
    const float maximum = *std::max_element(values.begin(), values.end());
    double sum = 0.0;

    for (float& value : values) {
        value = std::exp(value - maximum);
        sum += value;
    }

    for (float& value : values) {
        value = static_cast<float>(value / sum);
    }
}

void ensure_probabilities(std::vector<float>& values) {
    double sum = 0.0;
    bool already_probabilities = true;

    for (float value : values) {
        if (value < -1e-6f || value > 1.0001f) {
            already_probabilities = false;
            break;
        }
        sum += value;
    }

    if (already_probabilities && std::fabs(sum - 1.0) < 1e-3) {
        return;
    }

    softmax_in_place(values);
}

void print_prediction(const std::vector<float>& probabilities) {
    double cat_probability = 0.0;
    double dog_probability = 0.0;

    for (int index = kCatIndexStart; index <= kCatIndexEnd; ++index) {
        cat_probability += probabilities[static_cast<size_t>(index)];
    }
    for (int index = kDogIndexStart; index <= kDogIndexEnd; ++index) {
        dog_probability += probabilities[static_cast<size_t>(index)];
    }

    const double other_probability =
        std::max(0.0, 1.0 - cat_probability - dog_probability);

    std::string label;
    double probability = 0.0;

    if (cat_probability >= dog_probability) {
        if (cat_probability < 0.5) {
            label = "其他";
            probability = other_probability >= kStrongOtherProbability
                              ? other_probability
                              : cat_probability;
        } else {
            label = "猫";
            probability = cat_probability;
        }
    } else {
        if (dog_probability < 0.5) {
            label = "其他";
            probability = other_probability >= kStrongOtherProbability
                              ? other_probability
                              : dog_probability;
        } else {
            label = "狗";
            probability = dog_probability;
        }
    }

    std::cout << "类别: " << label << ", 概率: "
              << std::fixed << std::setprecision(4) << probability << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    HMODULE runtime_module = LoadLibraryA("onnxruntime.dll");
    if (runtime_module == nullptr) {
        std::cerr << "Failed to load onnxruntime.dll. Error: "
                  << GetLastError() << std::endl;
        return 1;
    }

    const auto get_api_base = reinterpret_cast<OrtGetApiBaseFn>(
        GetProcAddress(runtime_module, "OrtGetApiBase"));
    if (get_api_base == nullptr) {
        std::cerr << "Failed to resolve OrtGetApiBase." << std::endl;
        FreeLibrary(runtime_module);
        return 1;
    }

    const OrtApiBase* api_base = get_api_base();
    if (api_base == nullptr) {
        std::cerr << "OrtGetApiBase returned null." << std::endl;
        FreeLibrary(runtime_module);
        return 1;
    }

    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        std::cerr << "Failed to get ONNX Runtime API." << std::endl;
        FreeLibrary(runtime_module);
        return 1;
    }

    OrtEnv* env = nullptr;
    OrtSessionOptions* session_options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory_info = nullptr;

    auto release_runtime = [&]() {
        if (memory_info != nullptr) {
            api->ReleaseMemoryInfo(memory_info);
            memory_info = nullptr;
        }
        if (session != nullptr) {
            api->ReleaseSession(session);
            session = nullptr;
        }
        if (session_options != nullptr) {
            api->ReleaseSessionOptions(session_options);
            session_options = nullptr;
        }
        if (env != nullptr) {
            api->ReleaseEnv(env);
            env = nullptr;
        }
    };

    if (!check_status(
            api,
            api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CatDogCpp", &env),
            "CreateEnv")) {
        release_runtime();
        FreeLibrary(runtime_module);
        return 1;
    }

    if (!check_status(api, api->CreateSessionOptions(&session_options),
                      "CreateSessionOptions") ||
        !check_status(api, api->SetIntraOpNumThreads(session_options, 1),
                      "SetIntraOpNumThreads") ||
        !check_status(api, api->SetSessionGraphOptimizationLevel(
                              session_options, ORT_ENABLE_ALL),
                      "SetSessionGraphOptimizationLevel") ||
        !check_status(api,
                      api->CreateSession(env, kModelPath, session_options,
                                         &session),
                      "CreateSession") ||
        !check_status(api, api->CreateCpuMemoryInfo(
                              OrtArenaAllocator, OrtMemTypeDefault,
                              &memory_info),
                      "CreateCpuMemoryInfo")) {
        release_runtime();
        FreeLibrary(runtime_module);
        return 1;
    }

    std::vector<std::string> image_paths;
    if (argc > 1) {
        for (int index = 1; index < argc; ++index) {
            image_paths.emplace_back(argv[index]);
        }
    } else {
        image_paths = {
            "E:\\code\\edge\\test.jpg",
            "E:\\code\\edge\\test2.jpg",
            "E:\\code\\edge\\test3.jpg",
        };
    }

    const int64_t input_shape[] = {1, kInputHeight, kInputWidth, kInputChannels};

    for (const std::string& image_path : image_paths) {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* image_data =
            stbi_load(image_path.c_str(), &width, &height, &channels, 3);

        if (image_data == nullptr) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            continue;
        }

        std::vector<float> input_values =
            preprocess_image(image_data, width, height);
        stbi_image_free(image_data);

        OrtValue* input_tensor = nullptr;
        OrtValue* output_tensor = nullptr;
        OrtTensorTypeAndShapeInfo* output_info = nullptr;

        if (!check_status(api,
                          api->CreateTensorWithDataAsOrtValue(
                              memory_info, input_values.data(),
                              input_values.size() * sizeof(float), input_shape,
                              4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              &input_tensor),
                          "CreateTensorWithDataAsOrtValue")) {
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        if (!check_status(api,
                          api->Run(session, nullptr, kInputNames, &input_tensor,
                                   1, kOutputNames, 1, &output_tensor),
                          "Run")) {
            api->ReleaseValue(input_tensor);
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        if (!check_status(
                api, api->GetTensorTypeAndShape(output_tensor, &output_info),
                "GetTensorTypeAndShape")) {
            api->ReleaseValue(output_tensor);
            api->ReleaseValue(input_tensor);
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        size_t output_count = 0;
        if (!check_status(
                api, api->GetTensorShapeElementCount(output_info, &output_count),
                "GetTensorShapeElementCount")) {
            api->ReleaseTensorTypeAndShapeInfo(output_info);
            api->ReleaseValue(output_tensor);
            api->ReleaseValue(input_tensor);
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        if (output_count != 1000) {
            std::cerr << "Unexpected output count: " << output_count << std::endl;
            api->ReleaseTensorTypeAndShapeInfo(output_info);
            api->ReleaseValue(output_tensor);
            api->ReleaseValue(input_tensor);
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        void* output_data = nullptr;
        if (!check_status(api,
                          api->GetTensorMutableData(output_tensor, &output_data),
                          "GetTensorMutableData")) {
            api->ReleaseTensorTypeAndShapeInfo(output_info);
            api->ReleaseValue(output_tensor);
            api->ReleaseValue(input_tensor);
            release_runtime();
            FreeLibrary(runtime_module);
            return 1;
        }

        const float* output_values = static_cast<const float*>(output_data);
        std::vector<float> probabilities(output_values,
                                         output_values + output_count);
        ensure_probabilities(probabilities);
        print_prediction(probabilities);

        api->ReleaseTensorTypeAndShapeInfo(output_info);
        api->ReleaseValue(output_tensor);
        api->ReleaseValue(input_tensor);
    }

    release_runtime();
    FreeLibrary(runtime_module);
    return 0;
}