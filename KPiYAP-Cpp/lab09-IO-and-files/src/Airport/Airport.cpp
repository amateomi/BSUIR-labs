#include "Airport.hpp"

#include "../Menu/Menu.hpp"

///////////////////////////////////////////////////////////////////////////////
// Constructors and destructors ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

Airport::Airport(std::string name, int aircraftAmount, int passengersPerDay)
        : name(std::move(name)), aircraftAmount(aircraftAmount), passengersPerDay(passengersPerDay) {}

///////////////////////////////////////////////////////////////////////////////
// Getters and setters ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const std::string& Airport::getName() const {
    return name;
}

void Airport::setName(const std::string& newName) {
    Airport::name = newName;
}

int Airport::getAircraftAmount() const {
    return aircraftAmount;
}

void Airport::setAircraftAmount(int newAircraftAmount) {
    Airport::aircraftAmount = newAircraftAmount;
}

int Airport::getPassengersPerDay() const {
    return passengersPerDay;
}

void Airport::setPassengersPerDay(int newPassengersPerDay) {
    Airport::passengersPerDay = newPassengersPerDay;
}

///////////////////////////////////////////////////////////////////////////////
// Text I/O ///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::istream& operator>>(std::istream& source, Airport& airport) {
    std::cout << "Enter name:";
    source >> airport.name;

    // Validation check
    for (auto& item: airport.name) {
        if (!std::islower(item)) {
            throw std::runtime_error("Expected lower letters for name");
        }
    }

    std::cout << "Enter aircraft amount:";
    airport.aircraftAmount = readInt();

    std::cout << "Enter passengers per day:";
    airport.passengersPerDay = readInt();

    return source;
}

std::ostream& operator<<(std::ostream& target, const Airport& airport) {
    target << "name: " << airport.name << std::endl
           << "aircraft amount: " << airport.aircraftAmount << std::endl
           << "passengers per day: " << airport.passengersPerDay << std::endl;
    return target;
}

std::ifstream& operator>>(std::ifstream& source, Airport& airport) {
    try {
        // Read name
        source >> airport.name;
        // Validation check
        for (auto& item: airport.name) {
            if (!std::islower(item)) {
                throw std::runtime_error("Expected lower letters for name");
            }
        }

        // Read aircraftAmount
        source >> airport.aircraftAmount;

        // Read passengersPerDay
        source >> airport.passengersPerDay;
    } catch (std::ifstream::failure&) {
        source.clear();
        throw std::runtime_error("Invalid data in stream");
    }

    return source;
}

std::ofstream& operator<<(std::ofstream& target, const Airport& airport) {
    try {
        target << airport.name << ' '
               << airport.aircraftAmount << ' '
               << airport.passengersPerDay << '\n';
    } catch (std::ostream::failure&) {
        throw std::runtime_error("Unable to output data in file");
    }

    return target;
}

///////////////////////////////////////////////////////////////////////////////
// Text files handling ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Airport::readText(std::ifstream& file) {
    if (!file.is_open()) {
        throw std::runtime_error("File in closed");
    }

    char ch;

    try {
        // Read name
        name.clear();
        while ((ch = static_cast<char>(file.get())) != ' ') {
            name.push_back(ch);
        }

        // Read aircraftAmount
        aircraftAmount = 0;
        while ((ch = static_cast<char>(file.get())) != ' ') {
            // Check for digits
            if (!std::isdigit(ch)) {
                throw std::runtime_error("Expected digits for aircraftAmount");
            }
            aircraftAmount = aircraftAmount * 10 + (ch - '0');
        }

        // Read passengersPerDay
        passengersPerDay = 0;
        while ((ch = static_cast<char>(file.get())) != '\n') {
            // Check for digits
            if (!std::isdigit(ch)) {
                throw std::runtime_error("Expected digits for passengersPerDay");
            }
            passengersPerDay = passengersPerDay * 10 + (ch - '0');
        }
    } catch (std::ifstream::failure&) {
        throw std::runtime_error("Invalid data in file");
    }
}

void Airport::writeText(std::ofstream& file) {
    if (!file.is_open()) {
        throw std::runtime_error("File in closed");
    }

    try {
        // Write name
        for (auto& item: name) {
            file.put(item);
        }
        file.put(' ');

        // Write aircraftAmount
        for (auto& item: std::to_string(aircraftAmount)) {
            file.put(item);
        }
        file.put(' ');

        // Write passengersPerDay
        for (auto& item: std::to_string(passengersPerDay)) {
            file.put(item);
        }
        file.put('\n');
    } catch (std::ofstream::failure&) {
        throw std::runtime_error("Unable to write down data into file");
    }
}

///////////////////////////////////////////////////////////////////////////////
// Binary file handling ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void Airport::readBin(std::ifstream& file) {
    if (!file.is_open()) {
        throw std::runtime_error("File in closed");
    }

    try {
        // Read name size
        std::size_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));

        // Read name
        char *str = new char[size + 1];
        file.read(str, static_cast<std::streamsize>(size));
        str[size] = '\0';
        name.clear();
        name = str;
        delete[] str;

        // Read aircraftAmount
        file.read(reinterpret_cast<char *>(&aircraftAmount), sizeof(aircraftAmount));

        // Read passengersPerDay
        file.read(reinterpret_cast<char *>(&passengersPerDay), sizeof(passengersPerDay));
    } catch (std::ifstream::failure&) {
        throw std::runtime_error("Invalid data in file");
    }
}

void Airport::writeBin(std::ofstream& file) {
    if (!file.is_open()) {
        throw std::runtime_error("File in closed");
    }

    try {
        // Get size of the name
        std::size_t size = name.size();
        // Write size of string
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // Write name (string)
        file.write(name.c_str(), static_cast<std::streamsize>(name.size()));

        // Write aircraftAmount
        file.write(reinterpret_cast<const char *>(&aircraftAmount), sizeof(aircraftAmount));

        // Write passengersPerDay
        file.write(reinterpret_cast<const char *>(&passengersPerDay), sizeof(passengersPerDay));
    } catch (std::ofstream::failure&) {
        throw std::runtime_error("Unable to write down data into file");
    }
}
