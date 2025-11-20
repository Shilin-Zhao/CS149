#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#define TILE_X 16
#define TILE_Y 16
#define THREADS_PER_BLK (TILE_X * TILE_Y)


#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "cycleTimer.h"
#include "exclusiveScan.cu_inl"



#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    
    cudaDeviceTileCnt=NULL;
    cudaDeviceTileOffset=NULL;
    cudaDeviceCircleIndex=NULL;
    cudaDeviceTileCirclePair=NULL;

    cudaDeviceCircleCnt=NULL;
    cudaDeviceCircleOffset=NULL;
    
    useTileBased=false;

    cudaDeviceStart=NULL;
    cudaDeviceEnd=NULL;
    // cudaStreamCreate(&stream);
    // cudaDeviceTileOffset=NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        if(useTileBased){
            cudaFree(cudaDeviceTileCnt);
            cudaFree(cudaDeviceTileOffset);
            cudaFree(cudaDeviceCircleIndex);
        }
        else{
            cudaFree(cudaDeviceCircleCnt);
            cudaFree(cudaDeviceCircleOffset);
            cudaFree(cudaDeviceStart);
            cudaFree(cudaDeviceEnd);
            cudaFree(cudaDeviceTileCirclePair);
        }
    }
    // cudaStreamSynchronize(stream); 
    // cudaStreamDestroy(stream);
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);


    // malloc my data
    // cudaMalloc(&tileCnt, sizeof(int) * numCircles);
    useTileBased = numCircles <=2000;
    int tw=TILE_X, th=TILE_Y;
    int width = image->width;
    int height = image->height;
    int thc = (height+th-1)/th;
    int twc = (width+tw-1)/tw;
    int total_tile_count = thc * twc;
    if(useTileBased){
        cudaMalloc(&cudaDeviceTileCnt, sizeof(int) * total_tile_count);
        // printf("start malloc cudaDeviceTileOffset\n");
        cudaMalloc(&cudaDeviceTileOffset, sizeof(int) * total_tile_count);
        cudaMalloc(&cudaDeviceCircleIndex, sizeof(int) * 2048 * 1000);
        // printf("end malloc cudaDeviceTileOffset\n");
    }else{
        cudaMalloc(&cudaDeviceCircleCnt, sizeof(int)*numCircles);
        cudaMalloc(&cudaDeviceCircleOffset, sizeof(int)*numCircles);
        cudaMalloc(&cudaDeviceTileCirclePair, sizeof(int)*1024*1024*100);
        cudaMalloc(&cudaDeviceStart, sizeof(int)*total_tile_count);
        cudaMalloc(&cudaDeviceEnd, sizeof(int)*total_tile_count);
    }

    // else{
    //     cudaMalloc(&tileCnt, sizeof(int) * numCircles);
    // }

    // printf("width:%d, height%d\n", width, height);
    // cudaCheckError(cudaMalloc(&start, sizeof(int) * total_tile_count));
    // cudaCheckError(cudaMalloc(&end, sizeof(int) * total_tile_count));
    


    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

template<typename T>
void print_cuda_array(T* deviceArray, int N, std::string title, int maxSize=10){
    T* hostArray = new T[N];
    cudaMemcpy(hostArray, deviceArray, sizeof(T) * N, cudaMemcpyDeviceToHost);
    std::cout<<"**********" << title << "**********\n";
    std::cout<< "Count: " << N << ", array: ";
    for(int i=0; i<N && i<maxSize;i++){
        std::cout<<hostArray[i] << ", ";
    }
    std::cout<<std::endl;
    std::cout<<"**************************\n";
    delete[] hostArray;
}


__device__ __inline__ int
circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

__device__ __inline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}


__global__ void count_tiles_per_circle(int tw, int th, int* circleCnt){
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    int ci3 = 3*ci;

    if (ci >= cuConstRendererParams.numCircles)
        return;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[ci3]);
    float  rad = cuConstRendererParams.radius[ci];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
    
    short max_tx = (screenMaxX + tw - 1) / tw;
    short max_ty = (screenMaxY + th - 1) / th;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    circleCnt[ci] = 0;
    for(short tx=screenMinX/tw; tx<max_tx; tx++){
        for(short ty=screenMinY/th; ty<max_ty; ty++){
            if(circleInBox(p.x, p.y, rad, tx*tw*invWidth, (tx+1)*tw*invWidth, (ty+1)*th*invHeight, ty*th*invHeight)){
                circleCnt[ci] += 1;
            }
        }
    }

}

__global__ void tile_circle_pair(int tw, int th, int* pos, long long* output){
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    int ci3 = 3*ci;

    if (ci >= cuConstRendererParams.numCircles)
        return;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[ci3]);
    float  rad = cuConstRendererParams.radius[ci];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
    
    short max_tx = (screenMaxX + tw - 1) / tw;
    short max_ty = (screenMaxY + th - 1) / th;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    // 计算横向tile的数量
    int twc = (imageWidth + tw -1)/tw;
    int output_i = pos[ci];
    // long long l_ci = (long long)ci << 32;
    for(short tx=screenMinX/tw; tx<max_tx; tx++){
        for(short ty=screenMinY/th; ty<max_ty; ty++){
            float l = tx*tw*invWidth, r = (tx+1)*tw*invWidth, t = (ty+1)*th*invHeight, b = ty*th*invHeight;
            if(circleInBox(p.x, p.y, rad, l, r, t, b)){
                long long ti = ty * twc + tx;
                output[output_i++] = (ti << 32) + ci;
            }
        }
    }
}



__global__ void find_start_end_index(int N, long long* value, int* start, int* end){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    int ti = value[i] >> 32;
    if(i==0){
        start[ti] = 0;
    }
    else{
        int prev_ti = value[i-1] >> 32;
        if(ti != prev_ti){
            start[ti] = i;
            end[prev_ti] = i;
        }
    }
    if(i==N-1) end[ti] = N;
}


__device__ __inline__ void
shadeSnow(float rad, float2 pixelCenter, float3 p, float4* colorPtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    // float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read
    float4 existingColor = *colorPtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    *colorPtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ void
shadeNormal(float rad, float3 rgb, float2 pixelCenter, float3 p, float4* colorPtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    // float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    // float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    // simple: each circle has an assigned color
    // int index3 = 3 * circleIndex;
    // rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read
    float4 existingColor = *colorPtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    *colorPtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

template<bool isSnow>
__global__ void kernelRenderPixels() {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    short w = cuConstRendererParams.imageWidth;
    short h = cuConstRendererParams.imageHeight;
    if(i >=w*h) return;
    int numCircles = cuConstRendererParams.numCircles;
    int x = i%w;
    int y= i/w;
    float invWidth = 1.f / w;
    float invHeight = 1.f / h;
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y*w+x)]);

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f),
                                    invHeight * (static_cast<float>(y) + 0.5f));
    float4 existColor = *imgPtr;
    for(int ci=0;ci<numCircles;ci++){
        int ci3 = ci*3;
        float3 p = *(float3*)(&cuConstRendererParams.position[ci3]);
        float  rad = cuConstRendererParams.radius[ci];
        float3 rgb = *(float3*)&(cuConstRendererParams.color[ci3]);
        
        if(isSnow){
            shadeSnow(rad, pixelCenterNorm, p, &existColor);
        }
        else{
            shadeNormal(rad, rgb, pixelCenterNorm, p, &existColor);
        }
    }
    *imgPtr = existColor;
}

template<bool isSnow>
__global__ void render_tile(int* start, int* end, long long* pair){
    short w = cuConstRendererParams.imageWidth;
    short h = cuConstRendererParams.imageHeight;
    short tw=blockDim.x, th=blockDim.y;
    short tx=blockIdx.x, ty=blockIdx.y;
    short twc=gridDim.x;
    short x = tx * tw + threadIdx.x;
    short y = ty * th + threadIdx.y;
    if(x>=w||y>=h) return;
    int ti = ty*twc+tx;

    if(start[ti]<0) return;
    // printf("x: %d, y:%d\n", x, y);
    float invWidth = 1.f / w;
    float invHeight = 1.f / h;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f), invHeight * (static_cast<float>(y) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y * w + x)]);
    float4 existColor = *imgPtr;
    __shared__ float3 p_shared[THREADS_PER_BLK];
    __shared__ float rad_shared[THREADS_PER_BLK];
    __shared__ float3 rgb_shared[THREADS_PER_BLK];

    int e = end[ti];
    int locali = threadIdx.y * tw + threadIdx.x;
    for(int i=start[ti];i<e;i+=THREADS_PER_BLK){
        int batch_size = min(THREADS_PER_BLK, e-i);
        if(locali<batch_size){
            int ci = pair[i+locali];
            int ci3 = ci*3;
            p_shared[locali] = *(float3*)(&cuConstRendererParams.position[ci3]);
            rad_shared[locali] = cuConstRendererParams.radius[ci];
            rgb_shared[locali] = *(float3*)&(cuConstRendererParams.color[ci3]);
        }
        __syncthreads();
        
        for(int j=0;j<batch_size;j++){
            if(isSnow){
                shadeSnow(rad_shared[j], pixelCenterNorm, p_shared[j], &existColor);
            }
            else{
                shadeNormal(rad_shared[j], rgb_shared[j], pixelCenterNorm, p_shared[j], &existColor);
            }
        }
        __syncthreads();
        
    }
    *imgPtr = existColor;
}

template<bool isSnow>
__global__ void render_tile_pixel(){
    short w = cuConstRendererParams.imageWidth;
    short h = cuConstRendererParams.imageHeight;
    short tw=blockDim.x, th=blockDim.y;
    short tx=blockIdx.x, ty=blockIdx.y;
    short x = tx * tw + threadIdx.x;
    short y = ty * th + threadIdx.y;
    if(x>=w||y>=h) return;

    float invWidth = 1.f / w;
    float invHeight = 1.f / h;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f), invHeight * (static_cast<float>(y) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y * w + x)]);
    float4 existColor = *imgPtr;
    __shared__ int is_intersect[THREADS_PER_BLK];
    __shared__ float3 p_shared[THREADS_PER_BLK];
    __shared__ float rad_shared[THREADS_PER_BLK];
    __shared__ float3 rgb_shared[THREADS_PER_BLK];
    int locali = threadIdx.y * tw + threadIdx.x;
    int numCircles = cuConstRendererParams.numCircles;
    for(int i=0;i<numCircles;i+=THREADS_PER_BLK){
        int batch_size = min(THREADS_PER_BLK, numCircles-i);
        if(locali<batch_size){
            int ci = i+locali;
            int ci3 = ci*3;
            p_shared[locali] = *(float3*)(&cuConstRendererParams.position[ci3]);
            rad_shared[locali] = cuConstRendererParams.radius[ci];
            is_intersect[locali] = circleInBox(p_shared[locali].x, p_shared[locali].y, rad_shared[locali], tx*tw*invWidth, (tx+1)*tw*invWidth, (ty+1)*th*invHeight, ty*th*invHeight);
            if(is_intersect[locali]) rgb_shared[locali] = *(float3*)&(cuConstRendererParams.color[ci3]);
        }
        __syncthreads();
        
        for(int j=0;j<batch_size;j++){
            if(is_intersect[j]){
                if(isSnow){
                    shadeSnow(rad_shared[j], pixelCenterNorm, p_shared[j], &existColor);
                }
                else{
                    shadeNormal(rad_shared[j], rgb_shared[j], pixelCenterNorm, p_shared[j], &existColor);
                }
            }

        }
        __syncthreads();
        
    }
    *imgPtr = existColor;
}

void CudaRenderer::render_sorted_pairs(){

    bool isSnow = sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME;
    int width = image->width;
    int height = image->height;
    int blockSize = THREADS_PER_BLK;

    // tile width, tile height
    int tw=TILE_X , th=TILE_Y;
    // tile height count, tile width count
    int thc = (height+th-1)/th;
    int twc = (width+tw-1)/tw;
    int total_tile_count = thc * twc;
    dim3 blockDim(tw, th);
    dim3 gridDim(twc, thc);
    

    int circleThreads = (numCircles + blockSize -1) / blockSize;
    count_tiles_per_circle<<<circleThreads, blockSize>>>(tw, th, cudaDeviceCircleCnt);



    //使用exclusive scan 计算每个circle对应的起始位置
    // 0.1ms
    thrust::device_ptr<int> circle_cnt_ptr(cudaDeviceCircleCnt);
    thrust::device_ptr<int> circle_offset_ptr(cudaDeviceCircleOffset);
    thrust::exclusive_scan(circle_cnt_ptr, circle_cnt_ptr + numCircles, circle_offset_ptr);


    // 获取相交的tile数量
    // 0.03 实在不行可以考虑优化
    int lastCircle, prevCircle;
    cudaMemcpy(&lastCircle, cudaDeviceCircleCnt + numCircles-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&prevCircle, cudaDeviceCircleOffset+numCircles-1, sizeof(int), cudaMemcpyDeviceToHost);
    int totalPair = prevCircle + lastCircle;
    // long long* tileCirclePair;
    // 获取(tile_id, circle_id)的数据。
    // cudaMalloc(&tileCirclePair, sizeof(long long) * totalPair);

    
    // 0.04
    tile_circle_pair<<<circleThreads, blockSize>>>(tw, th, cudaDeviceCircleOffset, cudaDeviceTileCirclePair);

    // 0.45ms 
    thrust::device_ptr<long long> tile_circle_ptr(cudaDeviceTileCirclePair);
    thrust::sort(tile_circle_ptr, tile_circle_ptr+totalPair);

    
    cudaMemset(cudaDeviceStart, -1, sizeof(int) * total_tile_count);
    cudaMemset(cudaDeviceEnd, -1, sizeof(int) * total_tile_count);
    find_start_end_index<<<(totalPair + blockSize - 1)/blockSize,blockSize>>>(totalPair, cudaDeviceTileCirclePair, cudaDeviceStart, cudaDeviceEnd);

    
    if(isSnow){
        render_tile<true><<<gridDim, blockDim>>>(cudaDeviceStart, cudaDeviceEnd, cudaDeviceTileCirclePair);
    }
    else{
        render_tile<false><<<gridDim, blockDim>>>(cudaDeviceStart, cudaDeviceEnd, cudaDeviceTileCirclePair);
    }

    // cudaFree(tileCirclePair);
}


void CudaRenderer::render_pixel_parallel(){
    bool isSnow = sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME;
    int width = image->width;
    int height = image->height;
    int blockSize = THREADS_PER_BLK;
    if(isSnow){
        kernelRenderPixels<true><<<(width*height+blockSize-1)/blockSize, blockSize>>>();
    }
    else{
        kernelRenderPixels<false><<<(width*height+blockSize-1)/blockSize, blockSize>>>();
    }
}

__global__ void count_circles_per_tile(int tw, int th, int* tileCnt){
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    int ci3 = 3*ci;

    if (ci >= cuConstRendererParams.numCircles)
        return;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[ci3]);
    float  rad = cuConstRendererParams.radius[ci];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    // 在这种情况，如果取整，可能有时候会漏掉一些边界的，所以最好扩大一点边界。
    short minX = static_cast<short>(imageWidth * (p.x - rad))-1;
    short maxX = static_cast<short>(imageWidth * (p.x + rad))+2;
    short minY = static_cast<short>(imageHeight * (p.y - rad))-1;
    short maxY = static_cast<short>(imageHeight * (p.y + rad))+2;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;
    
    short max_tx = (screenMaxX + tw - 1) / tw;
    short max_ty = (screenMaxY + th - 1) / th;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int twc = (imageWidth + tw -1)/tw;
    for(int ty=screenMinY/th; ty<max_ty; ty++){
        for(int tx=screenMinX/tw; tx<max_tx; tx++){
            float l=tx*tw*invWidth, r=(tx+1)*tw*invWidth, t=(ty+1)*th*invHeight, b=ty*th*invHeight;
            if(circleInBox(p.x, p.y, rad, l, r, t, b)){
                // int ti = ty*twc+tx;
                atomicAdd(&tileCnt[ty*twc+tx], 1);
                // tileCnt[ci] += 1;
            }
        }
    }

}

__global__ void fill_circle_index(int N, int *tileOffset, int* circleIndex){


    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    bool isValidTi = ti<N;
    // if(ti>=N) return;

    // tile width, tile height
    short tw=TILE_X , th=TILE_Y;
    short w = cuConstRendererParams.imageWidth;
    short h = cuConstRendererParams.imageHeight;
    short twc = (w+tw-1)/tw;
    short tx = ti%twc, ty = ti/twc;
    float invWidth = 1.f / w;
    float invHeight = 1.f / h;

    int offset = isValidTi ? tileOffset[ti] : 0;
    float l=tx*tw*invWidth, r=(tx+1)*tw*invWidth, t=(ty+1)*th*invHeight, b=ty*th*invHeight;
    __shared__ float px[THREADS_PER_BLK], py[THREADS_PER_BLK], rad[THREADS_PER_BLK];
    int locali = threadIdx.x;
    int e = cuConstRendererParams.numCircles;
    for(int ci=0;ci<e;ci+=THREADS_PER_BLK){
        int batch_size = min(THREADS_PER_BLK, e-ci);
        int ci3 = (ci+locali)*3;
        if(locali<batch_size){
            px[locali] = cuConstRendererParams.position[ci3];
            py[locali] = cuConstRendererParams.position[ci3+1];
            rad[locali] = cuConstRendererParams.radius[ci+locali];
        }
        __syncthreads();
        if(isValidTi){
            for(int j=0;j<batch_size;j++){
                if(circleInBoxConservative(px[j], py[j], rad[j], l, r, t, b) && circleInBox(px[j], py[j], rad[j], l, r, t, b)){
                    circleIndex[offset++] = ci+j;
                }
            }
        }
        __syncthreads();

    }
}


template<bool isSnow>
__global__ void render_tile2(int* tileCnt, int* tileOffset, int* circleIndex){
    short w = cuConstRendererParams.imageWidth;
    short h = cuConstRendererParams.imageHeight;
    short tw=blockDim.x, th=blockDim.y;
    short tx=blockIdx.x, ty=blockIdx.y;
    short twc=gridDim.x;
    short x = tx * tw + threadIdx.x;
    short y = ty * th + threadIdx.y;
    if(x>=w||y>=h) return;
    int ti = ty*twc+tx;

    if(tileCnt[ti]<=0) return;
    // printf("x: %d, y:%d\n", x, y);
    float invWidth = 1.f / w;
    float invHeight = 1.f / h;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f), invHeight * (static_cast<float>(y) + 0.5f));
    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y * w + x)]);
    float4 existColor = *imgPtr;
    __shared__ float3 p_shared[THREADS_PER_BLK];
    __shared__ float rad_shared[THREADS_PER_BLK];
    __shared__ float3 rgb_shared[THREADS_PER_BLK];

    int e = tileCnt[ti] + tileOffset[ti];
    int locali = threadIdx.y * tw + threadIdx.x;
    for(int i=tileOffset[ti];i<e;i+=THREADS_PER_BLK){
        int batch_size = min(THREADS_PER_BLK, e-i);
        if(locali<batch_size){
            int ci = circleIndex[i+locali];
            rad_shared[locali] = cuConstRendererParams.radius[ci];
            int ci3 = ci*3;
            p_shared[locali] = *(float3*)(&cuConstRendererParams.position[ci3]);
            rgb_shared[locali] = *(float3*)&(cuConstRendererParams.color[ci3]);
        }
        __syncthreads();
        
        for(int j=0;j<batch_size;j++){
            if(isSnow){
                shadeSnow(rad_shared[j], pixelCenterNorm, p_shared[j], &existColor);
            }
            else{
                shadeNormal(rad_shared[j], rgb_shared[j], pixelCenterNorm, p_shared[j], &existColor);
            }
        }
        __syncthreads();
        
    }
    *imgPtr = existColor;
}

void CudaRenderer::render_tile_based(){
    bool isSnow = sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME;
    int width = image->width;
    int height = image->height;
    int blockSize = THREADS_PER_BLK;

    // tile width, tile height
    int tw=TILE_X , th=TILE_Y;
    // tile height count, tile width count
    int thc = (height+th-1)/th;
    int twc = (width+tw-1)/tw;
    int total_tile_count = thc * twc;
    dim3 blockDim(tw, th);
    dim3 gridDim(twc, thc);


    cudaMemset(cudaDeviceTileCnt, 0, sizeof(int) * total_tile_count);
    count_circles_per_tile<<<(numCircles + blockSize -1) / blockSize, blockSize>>>(tw, th, cudaDeviceTileCnt);
    
    
    thrust::device_ptr<int> tile_cnt_ptr(cudaDeviceTileCnt);
    thrust::device_ptr<int> tile_offset_ptr(cudaDeviceTileOffset);
    thrust::exclusive_scan(tile_cnt_ptr, tile_cnt_ptr + total_tile_count, tile_offset_ptr);
    
    
    fill_circle_index<<<(total_tile_count+blockSize-1)/blockSize, blockSize>>>(total_tile_count, cudaDeviceTileOffset, cudaDeviceCircleIndex);

    if(isSnow){
        render_tile2<true><<<gridDim, blockDim>>>(cudaDeviceTileCnt, cudaDeviceTileOffset, cudaDeviceCircleIndex);
    }
    else{
        render_tile2<false><<<gridDim, blockDim>>>(cudaDeviceTileCnt, cudaDeviceTileOffset, cudaDeviceCircleIndex);
    }
}



void
CudaRenderer::render() {

    if(numCircles < 10){
        render_pixel_parallel();
    }
    else if(useTileBased){
        render_tile_based();
    }
    else{
        render_sorted_pairs();
    }
    
    return;

}
