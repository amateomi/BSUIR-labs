#include "Accountant.hpp"

#include <cassert>

void Accountant::setSalaryEmployee(Employee &employee, Employee::Salary salary) {
    assert(salary >= 0 && "Negative value for salary");
    employee.m_salary = salary;
}