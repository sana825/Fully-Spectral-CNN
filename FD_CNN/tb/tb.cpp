#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "inference.h"

// Include your test image header
#include "/path/to/MNIST_TEST_IMAGES_64/4/test_00004.h"

#define CHECK_GOLDEN   1     // 1 = compare against EXPECTED_CLASS, 0 = skip check
#define EXPECTED_CLASS 4   // <-- replace with the correct class index for ca.h

// ============================================================================
// Class Labels for 10 Character Classes
// ============================================================================

const char* CLASS_LABELS[NUM_CLASSES] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

// ============================================================================
// Copy from 1D header array to 2D array
// ============================================================================

void copy_from_flat_array(const float* source,
                          float dest[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            dest[i][j] = source[i * IMAGE_WIDTH + j];
        }
    }
}

// ============================================================================
// Copy from 2D header array to 2D array (if needed)
// ============================================================================

void copy_from_2d_array(const float source[IMAGE_HEIGHT][IMAGE_WIDTH],
                       float dest[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

// ============================================================================
// Simple preprocessing for single character
// ============================================================================

void preprocess_single_character(float input[IMAGE_HEIGHT][IMAGE_WIDTH],
                                 float output[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    // Simple copy - assumes image is already preprocessed
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            output[i][j] = input[i][j];
        }
    }
}

// ============================================================================
// Display image statistics
// ============================================================================

void display_image_stats(float image[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    float min_val = image[0][0];
    float max_val = image[0][0];
    float sum = 0.0f;

    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            float val = image[i][j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }
    }

    float mean = sum / (IMAGE_HEIGHT * IMAGE_WIDTH);

    std::cout << "Image Statistics:" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
}

// ============================================================================
// Get character label from class index
// ============================================================================

const char* get_class_label(int class_idx) {
    if (class_idx >= 0 && class_idx < NUM_CLASSES) {
        return CLASS_LABELS[class_idx];
    }
    return "?";
}

// ============================================================================
// Main Testbench
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "============================================================" << std::endl;
    std::cout << "  SPECTRAL CNN - CHARACTER RECOGNITION TESTBENCH" << std::endl;
    std::cout << "  Model: 3-Layer SpCNN with 10 Character Classes" << std::endl;
    std::cout << "  Mode : C/RTL Cosimulation (single image)" << std::endl;
    std::cout << "============================================================" << std::endl << std::endl;

    static float input_image[IMAGE_HEIGHT][IMAGE_WIDTH];
    static float processed_image[IMAGE_HEIGHT][IMAGE_WIDTH];

    static fixed_t logits[NUM_CLASSES];
    static fixed_t probs[NUM_CLASSES];

    std::cout << "Loading embedded test image..." << std::endl;
    copy_from_flat_array(input_test_00004, input_image);

    display_image_stats(input_image);
    std::cout << std::endl;

    preprocess_single_character(input_image, processed_image);

    std::cout << "Running SpCNN inference (3 layers, 10 classes)..." << std::endl;
    spcnn_inference(processed_image, logits);

    int pred_class = get_prediction(logits);
    softmax(logits, probs);
    float confidence = float(probs[pred_class]) * 100.0f;

    // Display results
    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  RESULTS" << std::endl;
    std::cout << "============================================================" << std::endl << std::endl;

    std::cout << "Predicted Character: '" << get_class_label(pred_class)
              << "' (Class " << pred_class << ")" << std::endl;
    std::cout << "Confidence: " << confidence << "%" << std::endl << std::endl;

    // Top 5 predictions
    std::cout << "Top 5 Predictions:" << std::endl;
    std::vector<std::pair<float, int> > prob_pairs;
    for (int i = 0; i < NUM_CLASSES; i++) {
        prob_pairs.push_back(std::make_pair(float(probs[i]), i));
    }
    std::sort(prob_pairs.rbegin(), prob_pairs.rend());

    for (int i = 0; i < 5 && i < NUM_CLASSES; i++) {
        std::cout << "  " << (i+1) << ". '" << get_class_label(prob_pairs[i].second)
                 << "' (Class " << prob_pairs[i].second << "): "
                 << (prob_pairs[i].first * 100.0f) << "%" << std::endl;
    }

    // Raw logits for debugging (first 10 and last 10)
    std::cout << std::endl << "Sample Raw Logits:" << std::endl;
    std::cout << "  First 10: [";
    for (int i = 0; i < 10 && i < NUM_CLASSES; i++) {
        std::cout << logits[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "  Last 10:  [";
    int start = NUM_CLASSES - 10;
    if (start < 0) start = 0;
    for (int i = start; i < NUM_CLASSES; i++) {
        std::cout << logits[i];
        if (i < NUM_CLASSES - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    int retval = 0;

#if CHECK_GOLDEN
    if (pred_class == EXPECTED_CLASS) {
        std::cout << std::endl << "TEST PASSED: predicted class " << pred_class
                   << " matches expected class " << EXPECTED_CLASS << std::endl;
        retval = 0;
    } else {
        std::cout << std::endl << "TEST FAILED: predicted class " << pred_class
                   << " ('" << get_class_label(pred_class) << "')"
                   << " does not match expected class " << EXPECTED_CLASS
                   << " ('" << get_class_label(EXPECTED_CLASS) << "')" << std::endl;
        retval = 1;
    }
#else
    std::cout << std::endl << "NOTE: golden-reference check disabled "
                 "(CHECK_GOLDEN=0) - inference ran to completion." << std::endl;
#endif

    std::cout << "  Inference Complete!" << std::endl;

    return retval;
}