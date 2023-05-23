#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define CHANNELS 3
#define BLOCK_SIZE 32

const char* inputPath = "../res/input.png";
const char* outputPath = "../res/output.png";

float* loadImageData (const unsigned char* img, unsigned int size);
unsigned char* saveImageData (const float* img, unsigned int size);
__global__ void robertsFilter(const float* input, float* output, int width, int height, int mode);

int main() {
    int width;
    int height;
    int rgb;
    int mode;

    std::cout << "1. Default\n2. Only gx\n3. Only gy\n4. Light mode\nSELECT: ";
    std::cin >> mode;

    std::cout << "\nLoading image...\n";
    unsigned char* image = stbi_load(inputPath, &width, &height, &rgb, CHANNELS);
    float* hostInput = loadImageData(image, width * height * CHANNELS);
    float* hostOutput = (float*)malloc(width * height * sizeof(float));
    float* deviceInput;
    float* deviceOutput;

    unsigned int imageSize = width * height * sizeof(unsigned int);

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);
    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Processing image...\n";
    robertsFilter<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height, mode);
    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    std::cout << "Saving result...\n";
    stbi_write_png(outputPath, width, height, CHANNELS,
                   saveImageData(hostOutput, width * height), width * CHANNELS);

    stbi_image_free(image);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);

    std::cout << "Done!\n";
    return 0;
}

__global__ void robertsFilter(const float* input, float* output, int width, int height, int mode) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - 1 && y < height - 1) {
        float gx = input[y * width + x] - input[(y+1) * width + (x+1)];
        float gy = input[y * width + (x+1)] - input[(y+1) * width + x];

        if(mode == 2)
            gy = 0.0;
        if(mode == 3)
            gx = 0.0;

        double magnitude = sqrtf((float)(gx * gx + gy * gy));

        // Light mode
        if(mode == 4)
        {
            magnitude = 1 - magnitude;
        }

        output[y * width + x] = (float)(magnitude);
    }

}

float* loadImageData (const unsigned char* img, unsigned int size) {
    float* data = new float[size / CHANNELS];

    int sum = 0;

    for (unsigned int i = 0; i < size; ++i) {
        sum += img[i];
        if ((i + 1) % CHANNELS == 0) {
            data[i / CHANNELS] = (sum / 3.f) / 255.f;
            sum = 0;
        }
    }

    return data;
}

unsigned char* saveImageData (const float* img, unsigned int size) {
    unsigned char* data = new unsigned char[size * CHANNELS];

    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < CHANNELS; ++j) {
            data[i * CHANNELS + j] = img[i] * 255;
        }
    }

    return data;
}
