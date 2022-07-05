#include "menu.hpp"

#include <algorithm>

#include "Accountant.hpp"

//--------------------------------Main Options-------------------------------//

void addEmployee(std::list<Employee> &employees) {
    // Get field information for new employee:
    // Get name
    std::cout << "Enter name:";
    Employee::Name name = inputName();
    // Get salary
    std::cout << "Enter salary:";
    Employee::Salary salary = inputSalary();

    // Add information
    employees.emplace_back(name);
    Accountant::setSalaryEmployee(employees.back(), salary);

    std::cout << "New employee successfully added!\n";
}

void showEmployees(const std::list<Employee> &employees) {
    if (employees.empty()) {
        std::cout << "Database is empty.\n";
        return;
    }

    std::cout << "Employees information:\n";
    for (auto &item: employees) {
        printEmployee(item);
    }
}

void removeEmployee(std::list<Employee> &employees) {
    std::cout << "Enter employee ID:";
    Employee::Id idRemove = inputId();

    for (auto iterator = employees.begin(); iterator != employees.end(); iterator++) {
        if (iterator->getId() == idRemove) {
            std::cout << iterator->getName() << " deleted." << std::endl;
            employees.erase(iterator);
            return;
        }
    }

    std::cout << "No such ID in database." << std::endl;
}

void changeEmployeeInformation(std::list<Employee> &employees) {
    std::cout << "Enter employee ID:";
    Employee::Id idChange = inputId();

    for (auto &item: employees) {
        if (item.getId() == idChange) {
            std::cout << "Which field change?\n";
            displayEmployeeFields();
            switch (inputOption()) {
            case 1: // Id
                std::cout << "Enter ID:";
                if (!item.setId(inputId())) {
                    std::cout << "Your ID is already used in database!\n";
                }
                return;

            case 2: // Name
                std::cout << "Enter name:";
                item.setName(inputName());
                return;

            case 3: // Salary
                std::cout << "Enter salary:";
                Accountant::setSalaryEmployee(item, inputSalary());
                return;

            default:
                std::cout << "Error: option out of range (1 ... 3).\n";
                return;
            }
        }
    }

    std::cout << "No such ID in database." << std::endl;
}

//--------------------------------Output info--------------------------------//

void displayMenuOptions() {
    std::cout << "Menu options:\n" <<
            "1) Add employee\n" <<
            "2) Show employee database\n" <<
            "3) Remove employee using ID\n" <<
            "4) Change employee information\n" <<
            "0) Quite\n" <<
            '>';
}

void displayEmployeeFields() {
    std::cout << "1) Id\n"
            << "2) Name\n"
            << "3) Salary\n"
            << '>';
}

//--------------------------------Input info---------------------------------//

Option inputOption() {
    Option option;

    while (true) {
        std::cin >> option;

        // Validation check
        if (std::cin.good() && std::cin.peek() == '\n') {
            std::cin.ignore();
            return option;
        }

        std::cout << "Error: invalid option input, use only digits\n";

        std::cin.clear();
        std::cin.ignore(256, '\n');

        std::cout << "Enter option again:";
    }
}

Employee::Id inputId() {
    Employee::Id id;

    while (true) {
        std::cin >> id;

        if (std::cin.good() && std::cin.peek() == '\n' &&
            isValidId(id)) {
            std::cin.ignore();
            return id;
        }

        std::cout << "Error: invalid ID input, ID is not negative integer value\n";

        std::cin.clear();
        std::cin.ignore(256, '\n');

        std::cout << "Enter ID again:";
    }
}

Employee::Name inputName() {
    Employee::Name name;

    while (true) {
        std::getline(std::cin, name);

        // Validation check
        if (isValidName(name)) {
            return name;
        }

        std::cout << "Error: invalid name input, don't use digits\n";

        std::cout << "Enter name again:";
    }
}

Employee::Salary inputSalary() {
    Employee::Salary salary;

    while (true) {
        std::cin >> salary;

        // Validation check
        if (std::cin.good() && std::cin.peek() == '\n' &&
            isValidSalary(salary)) {
            std::cin.ignore();
            return salary;
        }

        std::cout << "Error: invalid salary input, salary is not negative integer value\n";

        std::cin.clear();
        std::cin.ignore(256, '\n');

        std::cout << "Enter salary again:";
    }
}

//---------------------------------Validation--------------------------------//

bool isValidId(const Employee::Id id) {
    return id >= 0;
}

bool isValidName(const Employee::Name &name) {
    return find_if(name.begin(), name.end(), isdigit) == name.end() && !name.empty();
}

bool isValidSalary(const Employee::Salary salary) {
    return salary >= 0;
}