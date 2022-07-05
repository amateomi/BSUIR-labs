#ifndef LAB2_SRC_MENU_HPP_
#define LAB2_SRC_MENU_HPP_

#include <iostream>
#include <list>

#include "Employee.hpp"

//--------------------------------User Types---------------------------------//

using Option = int;

//--------------------------------Main Options-------------------------------//

void addEmployee(std::list<Employee> &employees);

void showEmployees(const std::list<Employee> &employees);

void removeEmployee(std::list<Employee> &employees);

void changeEmployeeInformation(std::list<Employee> &employees);

//--------------------------------Output info--------------------------------//

void displayMenuOptions();

void displayEmployeeFields();

//--------------------------------Input info---------------------------------//

Option inputOption();

Employee::Id inputId();

Employee::Name inputName();

Employee::Salary inputSalary();

//---------------------------------Validation--------------------------------//

bool isValidId(Employee::Id id);

// Return true when no digits in name
bool isValidName(const Employee::Name &name);

// Return true if positive value
bool isValidSalary(Employee::Salary salary);

#endif //LAB2_SRC_MENU_HPP_
