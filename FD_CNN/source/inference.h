#ifndef INFERENCE_H
#define INFERENCE_H

#include <ap_fixed.h>
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <complex>

// Configuration constants
#define IMAGE_HEIGHT 64
#define IMAGE_WIDTH 64
#define IN_CHANNELS 1
#define NUM_CLASSES 10
#define NUM_LAYERS 3

// Layer configurations
#define LAYER1_KERNELS 2
#define LAYER2_KERNELS 4
#define LAYER3_KERNELS 16

// Maximum dimensions for buffer allocation
#define MAX_HEIGHT 64
#define MAX_WIDTH 64
#define MAX_CHANNELS 16


typedef ap_fixed<24, 13> fixed_t;
typedef std::complex<fixed_t> complex_fixed_t; 

// FFT Constants
#define PI 3.14159265358979323846

#define FIXED_TO_FLOAT(x) ((float)(x))

// Function declarations

void fft_1d(complex_fixed_t input[], complex_fixed_t output[], int N, bool inverse);

void fft_2d(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH], 
            complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

void fft_shift(complex_fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
               complex_fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]);

void spectral_relu(complex_fixed_t* data, int height, int width, int channels);

void spectral_pooling(complex_fixed_t* input, complex_fixed_t* output,
                      int in_height, int in_width, int channels);

void batch_norm(complex_fixed_t* data,
                const float gamma[], const float beta[],
                const float mean[], const float variance[],
                int height, int width, int channels,
                bool is_real);

void spectral_conv_layer(complex_fixed_t* input,
                         complex_fixed_t* output,
                         const float* weights_real, const float* weights_imag,
                         const float* bias_real, const float* bias_imag,
                         const float* bn_real_gamma, const float* bn_real_beta,
                         const float* bn_real_mean, const float* bn_real_var,
                         const float* bn_imag_gamma, const float* bn_imag_beta,
                         const float* bn_imag_mean, const float* bn_imag_var,
                         int height, int width, int in_channels, int out_channels);

void fully_connected(complex_fixed_t* input,
                    fixed_t output[NUM_CLASSES],
                    int height, int width, int channels);

void spcnn_inference(float input_image[IMAGE_HEIGHT][IMAGE_WIDTH],
                     fixed_t output_logits[NUM_CLASSES]);

void softmax(fixed_t input[NUM_CLASSES], fixed_t output[NUM_CLASSES]);

int get_prediction(fixed_t logits[NUM_CLASSES]);

// Utility functions for fixed-point arithmetic

// Complex multiplication
inline complex_fixed_t complex_mult(complex_fixed_t a, complex_fixed_t b) {
    #pragma HLS INLINE
    fixed_t real_part = a.real() * b.real() - a.imag() * b.imag();
    fixed_t imag_part = a.real() * b.imag() + a.imag() * b.real();
    return complex_fixed_t(real_part, imag_part);
}

// Fixed-point division with scaling
inline fixed_t fixed_div(fixed_t a, fixed_t b) {
    #pragma HLS INLINE
    if (b == 0) return (fixed_t)0;
    return a / b;
}

// Fast approximate square root for fixed-point
inline fixed_t fixed_sqrt(fixed_t x) {
    #pragma HLS INLINE
    if (x <= 0) return (fixed_t)0;
    return (fixed_t)hls::sqrt((float)x);
}

// Fixed-point exponential approximation for softmax
inline fixed_t fixed_exp(fixed_t x) {
    #pragma HLS INLINE
    return (fixed_t)hls::exp((float)x);
}

// Complex number magnitude for fixed-point
inline fixed_t complex_mag(complex_fixed_t c) {
    #pragma HLS INLINE
    fixed_t real_sq = c.real() * c.real();
    fixed_t imag_sq = c.imag() * c.imag();
    return fixed_sqrt(real_sq + imag_sq);
}

#endif // INFERENCE_H