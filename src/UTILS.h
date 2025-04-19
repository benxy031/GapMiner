#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

// Define color constants using ANSI escape codes
const std::string COLOR_GREEN = "\033[92m"; // Green
const std::string COLOR_YELLOW = "\033[93m"; // Yellow
const std::string COLOR_RED = "\033[91m"; // Red
const std::string COLOR_BLUE = "\033[94m"; // Blue (for DEBUG)
const std::string COLOR_RESET = "\033[0m"; // Reset to default color

// Define log levels with colors
#define LOG_E 0, COLOR_RED
#define LOG_W 1, COLOR_YELLOW
#define LOG_I 2, COLOR_GREEN
#define LOG_D 3, COLOR_BLUE

void log_str(const std::string& str, int status) {
    std::string colored_message;
    switch (status) {
        case 0:
            colored_message = COLOR_RED + str + COLOR_RESET;
            break;
        case 1:
            colored_message = COLOR_YELLOW + str + COLOR_RESET;
            break;
        case 2:
            colored_message = COLOR_GREEN + str + COLOR_RESET;
            break;
        case 3:
            colored_message = COLOR_BLUE + str + COLOR_RESET;
            break;
        default:
            colored_message = str;
    }
    std::cout << colored_message << std::endl;
}

#endif // UTILS_H