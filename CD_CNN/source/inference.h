#ifndef SPCNN_INFERENCE_H
#define SPCNN_INFERENCE_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

// ============================================================================
// Model Configuration
// ============================================================================
#define NUM_LAYERS 3
#define NUM_CLASSES 10
#define IMAGE_HEIGHT 64
#define IMAGE_WIDTH 64
#define IN_CHANNELS 1

#define LAYER1_KERNELS 2
#define LAYER2_KERNELS 4
#define LAYER3_KERNELS 16

#define LAYER1_KEEP_H 32   // int(64 * 0.5)
#define LAYER1_KEEP_W 32
#define LAYER2_KEEP_H 16   // int(32 * 0.5)
#define LAYER2_KEEP_W 16
#define LAYER3_KEEP_H 10   // int(16 * 0.65) = int(10.4)
#define LAYER3_KEEP_W 10

// Maximum dimensions for buffer allocation
#define MAX_HEIGHT 64
#define MAX_WIDTH 64
#define MAX_CHANNELS 16   // must cover the largest out_channels (layer 3)

#define N 64
#define LOG2_N 6
#define UF 4 

#define PI 3.14159265358979323846


typedef ap_fixed<17, 7, AP_RND, AP_SAT> fixed_t; 

struct cplx {
    fixed_t re, im;
    cplx(fixed_t r = fixed_t(0), fixed_t i = fixed_t(0)) : re(r), im(i) {}
    cplx operator+(const cplx& o) const { return cplx(re + o.re, im + o.im); }
    cplx operator-(const cplx& o) const { return cplx(re - o.re, im - o.im); }
    cplx operator*(const cplx& o) const {
        return cplx(re * o.re - im * o.im, re * o.im + im * o.re);
    }
};

// Conv layer 1: kept 32x32, 1 -> 2 channels
#define CONV1_WEIGHT_SIZE 2048     // 32*32*1*2
#define CONV1_BIAS_SIZE   2
#define BN1_SIZE           2

// Conv layer 2: kept 16x16, 2 -> 4 channels
#define CONV2_WEIGHT_SIZE 2048     // 16*16*2*4
#define CONV2_BIAS_SIZE   4
#define BN2_SIZE           4

// Conv layer 3: kept 10x10, 4 -> 16 channels
#define CONV3_WEIGHT_SIZE 6400     // 10*10*4*16
#define CONV3_BIAS_SIZE   16
#define BN3_SIZE           16 

#define DENSE_WEIGHTS_SIZE 16000  // 1600*94
#define DENSE_BIAS_SIZE    10

// ============================================================================
// Function Declarations
// ============================================================================
void dct_1d(fixed_t input[N], fixed_t output[N]);
void dct_2d(fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
            fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

void dct_relu(fixed_t* data, int height, int width, int channels);

void batch_norm(fixed_t* data,
                const fixed_t gamma[], const fixed_t beta[],
                const fixed_t mean[], const fixed_t variance[],
                int height, int width, int channels);

void spectral_conv_layer(fixed_t* input, fixed_t* output,
                         const fixed_t* weights,
                         const fixed_t* bias,
                         const fixed_t* bn_gamma, const fixed_t* bn_beta,
                         const fixed_t* bn_mean, const fixed_t* bn_var,
                         int in_height, int in_width, int in_channels,
                         int keep_h, int keep_w, int out_channels);

void fully_connected(fixed_t* input,
                    fixed_t output[NUM_CLASSES],
                    int height, int width, int channels);

void spcnn_inference(fixed_t input_image[IMAGE_HEIGHT][IMAGE_WIDTH],
                     fixed_t output_logits[NUM_CLASSES]);

void softmax(fixed_t input[NUM_CLASSES], fixed_t output[NUM_CLASSES]);
int get_prediction(fixed_t logits[NUM_CLASSES]);

#endif