#ifndef LAB2_SRC_EMPLOYEE_HPP_
#define LAB2_SRC_EMPLOYEE_HPP_

#include <vector>
#include <string>

class Employee {
    //------------------------------User Types-------------------------------//
public:

    using Id = int;
    using Name = std::string;
    using Salary = int;

    //--------------------------Statics and consts---------------------------//
private:

    // Store all used ID
    static std::vector<Id> m_idDatabase;

    static bool isUniqueId(Id id);

    //--------------------------------Fields---------------------------------//

    Id m_id{};
    Name m_name{};
    Salary m_salary{};

    //----------------------Constructors & Destructors-----------------------//
public:

    // Set unique ID for each new object
    // Set string argument to name, default = "none"
    // Set -1 to salary
    explicit Employee(const Name &name = "none");

    // Free one unique ID for reuse
    ~Employee();

    //---------------------------Getters & Setters---------------------------//
private:

    // Set unique ID for each new object
    void setId();

    // free ID for use
    void freeId() const;

    // Class Accountant use this function to set salary value
    void setSalary(Salary salary);

public:
    // Return false when id argument is not unique
    bool setId(Id id);

    [[nodiscard]] Id getId() const;

    [[nodiscard]] Name getName() const;

    void setName(Name name);

    [[nodiscard]] Salary getSalary() const;

    //--------------------------------Friends--------------------------------//

    // Output Employee data in std::cout
    friend void printEmployee(const Employee &employee);

    // Set salary field for Employee
    friend class Accountant;
};

#endif //LAB2_SRC_EMPLOYEE_HPP_
