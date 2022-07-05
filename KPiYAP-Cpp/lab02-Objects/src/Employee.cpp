#include "Employee.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>

//----------------------------Statics and consts-----------------------------//

std::vector<Employee::Id> Employee::m_idDatabase;

bool Employee::isUniqueId(Id id) {
    return std::find(m_idDatabase.begin(), m_idDatabase.end(), id) == m_idDatabase.end();
}

//------------------------Constructors & Destructors-------------------------//

Employee::Employee(const Name &name) {
    setId();
    setName(name);
    setSalary(-1);
}

Employee::~Employee() {
    freeId();
}

//-----------------------------Getters & Setters-----------------------------//

// Private:
void Employee::setId() {
    // Unique ID
    Employee::Id id = 1;

    // Find free ID
    while (!isUniqueId(id)) {
        ++id;
    }

    m_id = id;

    // Add new unique ID in database
    m_idDatabase.push_back(m_id);
}

void Employee::freeId() const {
    if (!m_idDatabase.empty()) {
        m_idDatabase.erase(std::find(m_idDatabase.begin(), m_idDatabase.end(), m_id));
    }
}

void Employee::setSalary(Employee::Salary salary) {
    m_salary = salary;
}

// Public:
bool Employee::setId(Id id) {
    if (isUniqueId(id)) {
        freeId();
        m_id = id;
        m_idDatabase.push_back(m_id);

        return true;
    } else {
        return false;
    }
}

Employee::Id Employee::getId() const {
    return m_id;
}

Employee::Name Employee::getName() const {
    return m_name;
}

void Employee::setName(Employee::Name name) {
    m_name = std::move(name);
}

Employee::Salary Employee::getSalary() const {
    return m_salary;
}

//----------------------------------Friends----------------------------------//

void printEmployee(const Employee &employee) {
    constexpr int COLUMN_SIZE = 10;
    std::cout << std::endl << std::setw(COLUMN_SIZE) << "Id: " << employee.getId()
            << std::endl << std::setw(COLUMN_SIZE) << "Name: " << employee.getName()
            << std::endl << std::setw(COLUMN_SIZE) << "Salary: " << employee.getSalary()
            << std::endl;
}
