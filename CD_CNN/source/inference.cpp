#include "inference.h"
#include "dct_mnist_weights.h"

static void fft_1d(cplx input[N], cplx output[N]) {

    if (N == 1) { output[0] = input[0]; return; }

    FFT_BITREV: for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE
    
    int j = 0, temp = i;
    for (int k = 0; k < LOG2_N; k++) {
        j = (j << 1) | (temp & 1);
        temp >>= 1;
    }
    output[i] = input[j];
}

    FFT_STAGES:
    for (int s = 1; s <= LOG2_N ; s++) {

        int m = 1 << s;
        int m2 = m >> 1;
        cplx w(fixed_t(1.0f), fixed_t(0.0f));

        float angle = -2.0f * (float)PI / m;
        cplx wm(fixed_t(hls::cos(angle)), fixed_t(hls::sin(angle)));

        FFT_GROUPS:
        for (int j = 0; j < m2; j++) {
            FFT_BUTTERFLY:
            for (int k = j; k < N; k += m) {
		cplx t = w * output[k + m2];
               
                #pragma HLS PIPELINE
                

                cplx u = output[k];
                output[k] = u + t;
                output[k + m2] = u - t;
            }
             w = w * wm;
             #pragma HLS BIND_OP variable=w op=mul impl=fabric
        }
    }
}

void dct_1d(fixed_t input[N], fixed_t output[N]) {
    
    cplx v[MAX_WIDTH];
    cplx V[MAX_WIDTH];


    DCT_REORDER:
    for (int n = 0; n < N / 2; n++) {
        #pragma HLS PIPELINE
        v[n]         = cplx(input[2 * n],     fixed_t(0));
        v[N - 1 - n] = cplx(input[2 * n + 1], fixed_t(0));
    }

    fft_1d(v, V);

    DCT_TWIDDLE:
    for (int k = 0; k < N; k++) {
        #pragma HLS PIPELINE
        
        float angle = -(float)PI * k / (2.0f * N);
        fixed_t cos_a = fixed_t(hls::cos(angle));
        fixed_t sin_a = fixed_t(hls::sin(angle));

        fixed_t re = V[k].re * cos_a - V[k].im * sin_a;

        float scale_f = (k == 0) ? hls::sqrt(1.0f / (4.0f * N))
                                  : hls::sqrt(1.0f / (2.0f * N));
        fixed_t scale = fixed_t(scale_f);

        output[k] = fixed_t(2) * re * scale;
    }
}


void dct_2d(fixed_t input[IMAGE_HEIGHT][IMAGE_WIDTH],
            fixed_t output[IMAGE_HEIGHT][IMAGE_WIDTH]) {

    fixed_t temp[IMAGE_HEIGHT][IMAGE_WIDTH];

    DCT_ROWS:
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        #pragma HLS LOOP_FLATTEN off
        fixed_t row_in [IMAGE_WIDTH];
        fixed_t row_out[IMAGE_WIDTH];

        for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE
            row_in[j] = input[i][j];
        }
        dct_1d(row_in, row_out);
        for (int j = 0; j < IMAGE_WIDTH; j++) {
             #pragma HLS PIPELINE
             temp[i][j] = row_out[j];
        }
    }

    // DCT along height
    DCT_COLS:
    for (int j = 0; j < IMAGE_WIDTH; j++) {
        #pragma HLS LOOP_FLATTEN off
        fixed_t col_in [IMAGE_HEIGHT];
        fixed_t col_out[IMAGE_HEIGHT];

        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            #pragma HLS PIPELINE
            col_in[i] = temp[i][j];
        }
        dct_1d(col_in, col_out);
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            #pragma HLS PIPELINE
            output[i][j] = col_out[i];
        }
    }
}


void format_dct_buffer(fixed_t dct_in[IMAGE_HEIGHT][IMAGE_WIDTH], 
                       fixed_t buffer_out[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS]) {
    
    FORMAT_LOOP_H: for (int i = 0; i < IMAGE_HEIGHT; i++) {
        FORMAT_LOOP_W: for (int j = 0; j < IMAGE_WIDTH; j++) {
            #pragma HLS PIPELINE
            buffer_out[i * IMAGE_WIDTH + j] = dct_in[i][j];
        }
    }
}


void dct_relu(fixed_t* data, int height, int width, int channels) {
    #pragma HLS INLINE off
 
    const float delta_base = 0.1f;
    
    DCT_RELU_H:
    for (int h = 0; h < height; h++) {
        DCT_RELU_W:
        for (int w = 0; w < width; w++) {
             
            float u = (float)h / (float)height;
            float v = (float)w / (float)width;
            float freq_dist = hls::sqrt(u * u + v * v);
            fixed_t delta = fixed_t(delta_base * (1.0f + freq_dist));

            DCT_RELU_C:
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE
                
                int idx = (h * width + w) * MAX_CHANNELS + c;
                fixed_t x = data[idx];
                fixed_t omega = (x < fixed_t(0)) ? fixed_t(-x) : x;
                fixed_t scale = delta / (omega + delta);
                data[idx] = x * scale;
            }
        }
    }
}


void batch_norm(fixed_t* data,
                const fixed_t gamma[], const fixed_t beta[],
                const fixed_t mean[], const fixed_t variance[],
                int height, int width, int channels) {
    #pragma HLS INLINE off

    const float epsilon = 1e-3f;

    BN_H:
    for (int h = 0; h < height; h++) {
        BN_W:
        for (int w = 0; w < width; w++) {
            BN_C:
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE

                int idx = (h * width + w) * MAX_CHANNELS + c;
                fixed_t value = data[idx];

                
                float var = variance[c];
                if (var < 0.0f) var = 0.0f;

                float std_f = hls::sqrt(var + epsilon);
                fixed_t std = fixed_t(std_f);

                fixed_t normalized = (value - fixed_t(mean[c])) / std;

                data[idx] = fixed_t(gamma[c]) * normalized + fixed_t(beta[c]);
            }
        }
    }
}


void spectral_conv_layer(fixed_t* input, fixed_t* output,
                         const fixed_t* weights,
                         const fixed_t* bias,
                         const fixed_t* bn_gamma, const fixed_t* bn_beta,
                         const fixed_t* bn_mean, const fixed_t* bn_var,
                         int in_height, int in_width, int in_channels,
                         int keep_h, int keep_w, int out_channels) {
    #pragma HLS INLINE off
	
    CONV_H:
    for (int h = 0; h < keep_h; h++) {
        CONV_W:
        for (int w = 0; w < keep_w; w++) {
            CONV_OUT:
            for (int out_c = 0; out_c < out_channels; out_c++) {

                fixed_t sum = fixed_t(0);
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    #pragma HLS PIPELINE
		            
                    int input_idx  = (h * in_width + w) * in_channels + in_c;
                    int weight_idx = ((h * keep_w + w) * in_channels + in_c) * out_channels + out_c;

                    sum += input[input_idx] * fixed_t(weights[weight_idx]);
                }

                sum += fixed_t(bias[out_c]);

                int output_idx = (h * keep_w + w) * out_channels + out_c;
                output[output_idx] = sum;
            }
        }
    }

    batch_norm(output, bn_gamma, bn_beta, bn_mean, bn_var, keep_h, keep_w, out_channels);
    dct_relu(output, keep_h, keep_w, out_channels);
}


void fully_connected(fixed_t* input,
                    fixed_t output[NUM_CLASSES],
                    int height, int width, int channels) {
    #pragma HLS INLINE off
    
    int feature_length = height * width * channels;

    fixed_t acc[NUM_CLASSES];
    #pragma HLS BIND_STORAGE variable=acc type=ram_1p impl=bram
    FC_INIT:
    for (int c = 0; c < NUM_CLASSES; c++) {
        acc[c] = fixed_t(dense_bias[c]);
    }

    FC_LOOP:
    for (int i = 0; i < feature_length; i++) {
        
        
        int h = i / (width * channels);
        int w = (i / channels) % width;
        int c = i % channels;
        int idx = (h * width + w) * MAX_CHANNELS + c;
        #pragma HLS BIND_OP variable=idx op=mul impl=fabric
        fixed_t val = input[idx];

        FC_CLASS:
        for (int out_c = 0; out_c < NUM_CLASSES; out_c++) {
            #pragma HLS PIPELINE
            acc[out_c] += val * fixed_t(dense_weights[i * NUM_CLASSES + out_c]);
        }
    }

    FC_OUT:
    for (int c = 0; c < NUM_CLASSES; c++) {
        #pragma HLS PIPELINE
        output[c] = (float)acc[c];
    }
}


// TOP FUNCTION ======================================================================

void spcnn_inference(fixed_t input_image[IMAGE_HEIGHT][IMAGE_WIDTH],
                     fixed_t output_logits[NUM_CLASSES]) {
   
    fixed_t dct_out[IMAGE_HEIGHT][IMAGE_WIDTH];
    fixed_t layer0_out[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];
    fixed_t layer1_out[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];
    fixed_t layer2_out[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];
    fixed_t layer3_out[MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS];

    #pragma HLS BIND_STORAGE variable=dct_out type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=layer0_out type=ram_2p impl=bram
	#pragma HLS BIND_STORAGE variable=layer1_out type=ram_2p impl=bram
	#pragma HLS BIND_STORAGE variable=layer2_out type=ram_2p impl=bram
	#pragma HLS BIND_STORAGE variable=layer3_out type=ram_2p impl=bram
   

    dct_2d(input_image, dct_out);
    format_dct_buffer(dct_out, layer0_out);
    

    spectral_conv_layer(layer0_out, layer1_out,
                       conv1_weights, conv1_bias,
                       bn1_gamma, bn1_beta, bn1_mean, bn1_var,
                       IMAGE_HEIGHT, IMAGE_WIDTH, 1,
                       LAYER1_KEEP_H, LAYER1_KEEP_W, LAYER1_KERNELS);

    spectral_conv_layer(layer1_out, layer2_out,
                       conv2_weights, conv2_bias,
                       bn2_gamma, bn2_beta, bn2_mean, bn2_var,
                       LAYER1_KEEP_H, LAYER1_KEEP_W, LAYER1_KERNELS,
                       LAYER2_KEEP_H, LAYER2_KEEP_W, LAYER2_KERNELS);

    spectral_conv_layer(layer2_out, layer3_out,
                       conv3_weights, conv3_bias,
                       bn3_gamma, bn3_beta, bn3_mean, bn3_var,
                       LAYER2_KEEP_H, LAYER2_KEEP_W, LAYER2_KERNELS,
                       LAYER3_KEEP_H, LAYER3_KEEP_W, LAYER3_KERNELS);

    fully_connected(layer3_out, output_logits,
                    LAYER3_KEEP_H, LAYER3_KEEP_W, LAYER3_KERNELS);
}

void softmax(fixed_t input[NUM_CLASSES], fixed_t output[NUM_CLASSES]) {
    fixed_t max_val = input[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float exp_vals[NUM_CLASSES];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {

        float shifted = (float)(input[i] - max_val);          
        exp_vals[i] = hls::exp(shifted);
        sum = sum + exp_vals[i];                       
    }

    for (int i = 0; i < NUM_CLASSES; i++) {

        float normalized = exp_vals[i] / sum;          
        output[i] = fixed_t(normalized);             
    }
}

int get_prediction(fixed_t logits[NUM_CLASSES]) {

    int max_idx = 0;
    fixed_t max_val = logits[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        
        if (logits[i] > max_val) {   
            max_val = logits[i];      
            max_idx = i;
        }
    }

    return max_idx;
}