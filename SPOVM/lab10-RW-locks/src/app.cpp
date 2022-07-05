#include "app.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

#include "record.hpp"

App::App(const std::filesystem::path &table_path)
    : table_(table_path) {}

void App::run() {
  int option;
  while (true) {
    std::cout << "1) - Add record\n"
              << "2) - Delete record\n"
              << "3) - Get record\n"
              << "4) - Put record\n"
              << "5) - Get primary\n"
              << "0) - Exit\n"
              << '>';

    std::cin >> option;
    while (std::cin.fail() || option != std::clamp(option, 0, 5)) {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      std::cout << "Invalid input\n";
      std::cin >> option;
    }

    switch (option) {
      case 1:
        addRecord();
        break;

      case 2:
        delRecord();
        break;

      case 3:
        getRecord();
        break;

      case 4:
        putRecord();
        break;

      case 5:
        getPrimary();
        break;

      default:
        exit(EXIT_SUCCESS);
    }
  }
}

void App::addRecord() {
  std::cout << "Enter record in format:\n"
            << prod::Record::INPUT_FORMAT << '\n'
            << '>';

  prod::Record record;
  try {
    std::cin >> record;
    if (std::cin.fail()) {
      throw std::runtime_error("Invalid input");
    }

    table_.addRecord(record);

  } catch (std::exception &exception) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << exception.what() << '\n';
  }
}

void App::delRecord() {
  std::cout << "Enter record ID:\n"
            << '>';

  int id;
  try {
    std::cin >> id;
    if (std::cin.fail() || id < 0) {
      throw std::runtime_error("Invalid input");
    }

    table_.delRecord(id);

  } catch (std::exception &exception) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << exception.what() << '\n';
  }
}

void App::getRecord() {
  std::cout << "Enter record ID:\n"
            << '>';

  int id;
  try {
    std::cin >> id;
    if (std::cin.fail() || id < 0) {
      throw std::runtime_error("Invalid input");
    }

    auto record = table_.getRecord(id);
    std::cout << prod::Record::TABLE_FORMAT << '\n'
              << record;

  } catch (std::exception &exception) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << exception.what() << '\n';
  }
}

void App::putRecord() {
  std::cout << "Enter record ID:\n"
            << '>';

  int id;
  try {
    std::cin >> id;
    if (std::cin.fail() || id < 0) {
      throw std::runtime_error("Invalid input");
    }

    std::cout << "Enter record in format:\n"
              << prod::Record::INPUT_FORMAT << '\n'
              << '>';

    prod::Record record;
    std::cin >> record;
    if (std::cin.fail()) {
      throw std::runtime_error("Invalid input");
    }

    record.id = id;

    table_.putRecord(record);

  } catch (std::exception &exception) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << exception.what() << '\n';
  }
}

void App::getPrimary() const {
  std::cout << "1) - Name\n"
            << "2) - Code\n"
            << '>';

  int option;
  try {
    std::cin >> option;
    if (std::cin.fail() || option != std::clamp(option, 1, 2)) {
      throw std::runtime_error("Invalid input");
    }

    // Name
    if (option == 1) {
      std::cout << "Enter name:\n"
                << '>';

      std::string name;
      std::cin >> name;
      if (std::cin.fail() || name.empty() || name.length() > 16) {
        throw std::runtime_error("Invalid input");
      }

      auto ids = std::get<std::list<int>>(table_.getPrimary(name));

      std::cout << "IDs:\n";
      for (const auto &id : ids) {
        std::cout << id << " ";
      }
      std::cout << '\n';

      // Code
    } else if (option == 2) {
      std::cout << "Enter code:\n"
                << '>';

      int code;
      std::cin >> code;
      if (std::cin.fail() || code < 0) {
        throw std::runtime_error("Invalid input");
      }

      auto id = std::get<int>(table_.getPrimary(code));
      std::cout << "ID: " << id << '\n';
    }

  } catch (std::exception &exception) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::cout << exception.what() << '\n';
  }
}
