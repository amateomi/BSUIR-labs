#include "Menu.hpp"

void eraseFiles(std::ofstream& target) {
    target.open(text, std::ios::trunc);
    target.close();
    target.open(textBin, std::ios::trunc);
    target.close();
    target.open(bin, std::ios::trunc);
    target.close();
}

///////////////////////////////////////////////////////////////////////////////
// Output /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::pair<int, int> showMainMenu() {
    std::cout << "Options:\n"
              << "1) - Input record to file/copy from file to file\n"
              << "2) - Display content form all files\n"
              << "3) - Search in files with key\n"
              << "4) - Search records in specific range\n"
              << "0) - Quit\n"
              << '>';
    return std::pair<int, int>{0, 4};
}

std::pair<int, int> showInputMenu() {
    std::cout << "Options:\n"
              << "1) - Input record to file\n"
              << "2) - Copy from file to file\n"
              << '>';
    return std::pair<int, int>{1, 2};
}

std::pair<int, int> showFileNames() {
    std::cout << "Options:\n"
              << "1) - " << text << '\n'
              << "2) - " << textBin << '\n'
              << "3) - " << bin << '\n'
              << '>';
    return std::pair<int, int>{1, 3};
}

std::pair<int, int> showAirportFields() {
    std::cout << "Options:\n"
              << "1) - name\n"
              << "2) - aircraft amount\n"
              << "3) - passengers per day\n"
              << '>';
    return std::pair<int, int>{1, 3};
}

void writeRecord(int type, std::ofstream& target, Airport& record) {
    switch (type) {
        case 1:
            target << record;
            break;

        case 2:
            record.writeText(target);
            break;

        case 3:
            record.writeBin(target);
            break;

        default:
            throw std::runtime_error("Unexpected file type in writeRecord()");
    }
}

void readRecord(int type, std::ifstream& source, Airport& record) {
    source.peek();
    if (!source.eof()) {
        switch (type) {
            case 1:
                source >> record;
                break;

            case 2:
                record.readText(source);
                break;

            case 3:
                record.readBin(source);
                break;

            default:
                throw std::runtime_error("Unexpected file type in readRecord()");
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Input //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int getMenuOption(std::pair<int, int>& range) {
    int value = readInt();
    // Range check
    if (value < range.first || range.second < value) {
        throw std::runtime_error("Value out of range");
    }
    return value;
}

int readInt() {
    int value;
    std::cin >> value;
    if (std::cin.fail() || std::cin.peek() != '\n') {
        std::cin.clear();
        std::cin.ignore(10, '\n');
        throw std::runtime_error("Read int failed");
    }
    return value;
}

///////////////////////////////////////////////////////////////////////////////
// Menu functions /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void inputRecords(std::ofstream& target) {
    Airport record;

    std::cout << "Select target file:" << std::endl;
    auto options = showFileNames();
    auto targetFileType = getMenuOption(options);
    openOutputStream(targetFileType, target);

    options = showInputMenu();
    switch (getMenuOption(options)) {
        case 1: // Input record to file
            // Get data
            while (true) {
                try {
                    std::cout << "Enter airport record:" << std::endl;
                    std::cin >> record;
                    break;
                } catch (std::runtime_error& error) {
                    std::cerr << "Error: " << error.what() << std::endl;
                    std::cin.clear();
                    std::cin.ignore(1000, '\n');
                }
            }

            // Add data
            writeRecord(targetFileType, target, record);
            std::cout << record << "added to file" << std::endl;
            break;

        case 2: // Copy from file to file
            std::ifstream source;
            std::list<Airport> listAirports;

            std::cout << "Select source file:" << std::endl;
            options = showFileNames();
            auto sourceFileType = getMenuOption(options);

            openInputStream(sourceFileType, source);

            // Get all records
            readRecord(sourceFileType, source, record);
            while (!source.eof()) {
                listAirports.push_back(record);
                readRecord(sourceFileType, source, record);
            }
            source.close();

            for (auto& item: listAirports) {
                writeRecord(targetFileType, target, item);
            }

            std::cout << "Copying succeed" << std::endl;
            break;
    }

    target.close();
}

void displayFiles(std::ifstream& source) {
    Airport record;
    std::list<Airport> listAirports;

    for (int i = 1; i <= 3; ++i) {
        // Read file content
        openInputStream(i, source);
        readRecord(i, source, record);
        while (!source.eof()) {
            listAirports.push_back(record);
            readRecord(i, source, record);
        }
        source.close();

        // Display records
        std::cout << "File " << i << std::endl;
        for (auto& item: listAirports) {
            std::cout << item;
        }
        std::cout << std::endl;
        listAirports.clear();
    }
}

void searchWithKey(std::ifstream& source) {
    std::string keyName;
    int keyAircraftAmount;
    int keyPassengersPerDSay;

    // Get key
    std::cout << "Select key type:" << std::endl;
    auto options = showAirportFields();
    int keyType = getMenuOption(options);
    std::cout << "Enter key:" << std::endl;
    switch (keyType) {
        case 1: // name
            std::cin >> keyName;
            std::cin.ignore(1000, '\n');
            break;

        case 2: // aircraft amount
            keyAircraftAmount = readInt();
            break;

        case 3: // passengers per day
            keyPassengersPerDSay = readInt();
            break;

        default:
            throw std::runtime_error("Unexpected file type in searchWithKey()");
    }

    Airport record;
    // Search match in all files
    for (int i = 1; i <= 3; ++i) {
        // Read file content
        openInputStream(i, source);
        readRecord(i, source, record);
        std::cout << "File " << i << std::endl;
        while (!source.eof()) {
            switch (keyType) {
                case 1: // name
                    if (keyName == record.getName()) {
                        std::cout << record;
                    }
                    break;

                case 2: // aircraft amount
                    if (keyAircraftAmount == record.getAircraftAmount()) {
                        std::cout << record;
                    }
                    break;

                case 3: // passengers per day
                    if (keyPassengersPerDSay == record.getPassengersPerDay()) {
                        std::cout << record;
                    }
                    break;
            }
            readRecord(i, source, record);
        }
        source.close();
    }
}

void searchInRange(std::ifstream& source) {
    std::pair<std::string, std::string> rangeName;
    std::pair<int, int> rangeAircraftAmount;
    std::pair<int, int> rangePassengersPerDSay;

    // Get range
    std::cout << "Select range type:" << std::endl;
    auto options = showAirportFields();
    int rangeType = getMenuOption(options);
    switch (rangeType) {
        case 1: // name
            std::cout << "Enter first string border:" << std::endl;
            std::cin >> rangeName.first;
            std::cin.ignore(1000, '\n');
            std::cout << "Enter second string border:" << std::endl;
            std::cin >> rangeName.second;
            std::cin.ignore(1000, '\n');
            // Set min value as first
            if (rangeName.first > rangeName.second) {
                std::swap(rangeName.first, rangeName.second);
            }
            break;

        case 2: // aircraft amount
            std::cout << "Enter first int border:" << std::endl;
            rangeAircraftAmount.first = readInt();
            std::cout << "Enter second int border:" << std::endl;
            rangeAircraftAmount.second = readInt();
            // Set min value as first
            if (rangeAircraftAmount.first > rangeAircraftAmount.second) {
                std::swap(rangeAircraftAmount.first, rangeAircraftAmount.second);
            }
            break;

        case 3: // passengers per day
            std::cout << "Enter first int border:" << std::endl;
            rangePassengersPerDSay.first = readInt();
            std::cout << "Enter second int border:" << std::endl;
            rangePassengersPerDSay.second = readInt();
            // Set min value as first
            if (rangePassengersPerDSay.first > rangePassengersPerDSay.second) {
                std::swap(rangePassengersPerDSay.first, rangePassengersPerDSay.second);
            }
            break;

        default:
            throw std::runtime_error("Unexpected file type in searchWithKey()");
    }

    Airport record;
    // Search match in all files
    for (int i = 1; i <= 3; ++i) {
        // Read file content
        openInputStream(i, source);
        readRecord(i, source, record);
        std::cout << "File " << i << std::endl;
        while (!source.eof()) {
            switch (rangeType) {
                case 1: // name
                    if (rangeName.first <= record.getName()
                        && record.getName() <= rangeName.second) {
                        std::cout << record;
                    }
                    break;

                case 2: // aircraft amount
                    if (rangeAircraftAmount.first <= record.getAircraftAmount()
                        && record.getAircraftAmount() <= rangeAircraftAmount.second) {
                        std::cout << record;
                    }
                    break;

                case 3: // passengers per day
                    if (rangePassengersPerDSay.first <= record.getPassengersPerDay()
                        && record.getPassengersPerDay() <= rangePassengersPerDSay.second) {
                        std::cout << record;
                    }
                    break;
            }
            readRecord(i, source, record);
        }
        source.close();
    }
}

///////////////////////////////////////////////////////////////////////////////
// File handling //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void openOutputStream(int type, std::ofstream& file) {
    switch (type) {
        case 1:
            file.open(text, std::ios::app);
            break;

        case 2:
            file.open(textBin, std::ios::binary | std::ios::app);
            break;

        case 3:
            file.open(bin, std::ios::binary | std::ios::app);
            break;

        default:
            throw std::runtime_error("Unexpected file type in openOutputStream()");
    }
}

void openInputStream(int type, std::ifstream& file) {
    switch (type) {
        case 1:
            file.open(text);
            break;

        case 2:
            file.open(textBin, std::ios::binary);
            break;

        case 3:
            file.open(bin, std::ios::binary);
            break;

        default:
            throw std::runtime_error("Unexpected file type in openOutputStream()");
    }
}
