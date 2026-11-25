#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include "inference.h"

// ---- Edit these for cosim run ----
static const char* HARD_CODED_PATH  = "/path/to/MNIST_TEST_IMAGES_64/4/test_00004.h";
static const int   HARD_CODED_CLASS = 4; // -1 if unknown / don't check

bool parse_h_file(const std::string& path, fixed_t pixels[4096]) {
    std::ifstream f(path.c_str());
    if (!f.is_open()) {
        std::cerr << "ERROR: could not open " << path << "\n";
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    size_t open  = content.find('{');
    size_t close = content.find('}', open);
    if (open == std::string::npos || close == std::string::npos) {
        std::cerr << "ERROR: no {...} array found in " << path << "\n";
        return false;
    }

    std::string data = content.substr(open + 1, close - open - 1);

    int idx = 0;
    std::istringstream token_stream(data);
    std::string token;
    while (std::getline(token_stream, token, ',') && idx < 4096) {
        size_t start = token.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = token.find_last_not_of(" \t\r\nf");
        if (end == std::string::npos) continue;
        std::string val_str = token.substr(start, end - start + 1);
        if (val_str.empty()) continue;
        pixels[idx++] = (float)atof(val_str.c_str());
    }

    if (idx != 4096) {
        std::cerr << "ERROR: expected 4096 values, parsed " << idx << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {

    std::string in_path = (argc > 1) ? argv[1] : HARD_CODED_PATH;
    int expected_class   = (argc > 2) ? atoi(argv[2]) : HARD_CODED_CLASS;

    std::cout << "============================================================\n"
              << "  Discrete Cosine Domain CNN\n"
              << "============================================================\n";
    std::cout << "Input file : " << in_path << "\n\n";

    // ---- Load the one image ----
    fixed_t pixels[4096];
    if (!parse_h_file(in_path, pixels)) {
        std::cerr << "FAILED to load input, aborting.\n";
        return 1;
    }

    fixed_t img[IMAGE_HEIGHT][IMAGE_WIDTH];
    for (int i = 0; i < IMAGE_HEIGHT; i++)
        for (int j = 0; j < IMAGE_WIDTH; j++)
            img[i][j] = pixels[i * IMAGE_WIDTH + j];

    std::cout << "---- INPUT (" << IMAGE_HEIGHT << " x " << IMAGE_WIDTH
              << ") ----\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            std::cout << img[i][j];
            if (j != IMAGE_WIDTH - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ---- Single DUT call 
    fixed_t logits[NUM_CLASSES];
    spcnn_inference(img, logits);

    int pred_class = get_prediction(logits);

    // ---- Dump every output logit ----
    std::cout << "---- OUTPUT LOGITS (" << NUM_CLASSES << ") ----\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        std::cout << "  logits[" << std::setw(2) << c << "] = "
                  << std::setw(12) << logits[c];
        if (c == pred_class) std::cout << "   <-- argmax";
        std::cout << "\n";
    }

    std::cout << "\n============================================================\n"
              << "  RESULT\n"
              << "============================================================\n"
              << "  Predicted class : " << pred_class << "\n";

    if (expected_class >= 0) {
        std::cout << "  Expected class  : " << expected_class << "\n"
                   << "  Match           : "
                   << (pred_class == expected_class ? "PASS" : "FAIL") << "\n";
    } else {
        std::cout << "  Expected class  : (not provided)\n";
    }
    std::cout << "============================================================\n";

    if (expected_class >= 0 && pred_class != expected_class) return 1;
    return 0;
}