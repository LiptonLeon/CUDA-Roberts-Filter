#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define CHANNELS 3
#define BLOCK_SIZE 32

const char* inputPath = "../res/input2.png";
const char* outputPath = "../res/output.png";

unsigned int* loadImageData (const unsigned char* img, unsigned int size);
unsigned char* saveImageData (const unsigned int* img, unsigned int size);
__global__ void robertsFilter(const unsigned int* input, unsigned int* output, int width, int height);

int main() {
    int width;
    int height;
    int rgb;

    unsigned char* image = stbi_load(inputPath, &width, &height, &rgb, CHANNELS);
    unsigned int* hostInput = loadImageData(image, width * height * CHANNELS);
    unsigned int* hostOutput = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    unsigned int* deviceInput;
    unsigned int* deviceOutput;

    unsigned int imageSize = width * height * sizeof(unsigned int);

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);
    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    robertsFilter<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height);
    //cudaMemcpy(deviceInput, deviceOutput, imageSize, cudaMemcpyDeviceToDevice);

    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_png(outputPath, width, height, CHANNELS,
                   saveImageData(hostOutput, width * height), width * CHANNELS);

    stbi_image_free(image);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);

    return 0;
}

__global__ void robertsFilter(const unsigned int* input, unsigned int* output, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width - 1 && y < height - 1) {
        unsigned int gx = input[(y+1) * width + x+1] - input[y * width + x];
        unsigned int gy = input[(y+1) * width + x] - input[y * width + x+1];
        double magnitude = sqrtf((float)(gx * gx + gy * gy));

        output[y * width + x] = (unsigned char)(magnitude);
    }
}

unsigned int* loadImageData (const unsigned char* img, unsigned int size) {
    unsigned int* data = new unsigned int[size / CHANNELS];

    int sum = 0;

    for (unsigned int i = 0; i < size; ++i) {
        sum += img[i];
        if ((i + 1) % CHANNELS == 0) {
            data[i / CHANNELS] = sum > 0 ? 1 : 0;
            sum = 0;
        }
    }

    return data;
}

unsigned char* saveImageData (const unsigned int* img, unsigned int size) {
    unsigned char* data = new unsigned char[size * CHANNELS];

    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < CHANNELS; ++j) {
            data[i * CHANNELS + j] = img[i] * 255;
        }
    }

    return data;
}
