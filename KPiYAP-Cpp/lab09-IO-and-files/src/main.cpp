#include "Menu/Menu.hpp"

int main() {
    std::ofstream target;
    std::ifstream source;

    eraseFiles(target);

    while (true) {
        auto options = showMainMenu();
        try {
            switch (getMenuOption(options)) {
                case 1: // Input record to file/copy from file to file
                    inputRecords(target);
                    break;

                case 2: // Display content form all files
                    displayFiles(source);
                    break;

                case 3: // Search in files with key
                    searchWithKey(source);
                    break;

                case 4: // Search records in specific range
                    searchInRange(source);
                    break;

                case 0: // Quit
                    exit(EXIT_SUCCESS);
            }
        } catch (std::exception& error) {
            if (target.is_open()) {
                target.close();

            }
            if (source.is_open()) {
                source.close();

            }
            std::cerr << "Error: " << error.what() << std::endl;
        }
        system("pause>0");
        system("cls");
    }
}