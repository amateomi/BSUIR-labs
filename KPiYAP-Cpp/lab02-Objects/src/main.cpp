#include <iostream>
#include <list>

// Classes
#include "Employee.hpp"
#include "Accountant.hpp"

#include "menu.hpp"

int main() {
    std::list<Employee> employees;

    while (true) {
        displayMenuOptions();
        switch (inputOption()) {
            // Add employee
        case 1:
            addEmployee(employees);
            break;

            // Show employee database
        case 2:
            showEmployees(employees);
            break;

            // Remove employee
        case 3:
            removeEmployee(employees);
            break;

            // Change employee information
        case 4:
            changeEmployeeInformation(employees);
            break;

            // Quite
        case 0:
            return 0;

            // No such option
        default:
            std::cout << "Error: option out of range (0 ... 4).\n";
            break;
        }

        std::cout << std::endl;
    }
}