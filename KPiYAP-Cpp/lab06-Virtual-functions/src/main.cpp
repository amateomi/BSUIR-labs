#include <iostream>

#include "Firearm/Firearm.hpp"
#include "ColdSteelArms/ColdSteelArms.hpp"

// Additional task functions
Weapon::Name inputName();
int inputInt();

int main() {
    constexpr int rangedWeapon  = 0;
    constexpr int firearm       = 1;
    constexpr int coldSteelArms = 2;
    
    constexpr int totalPointers = 3;
    
    // Create array of pointers to Weapon
    auto array = new Weapon *[totalPointers];
    
    //----------------------------Additional task----------------------------//
    std::cout << "Enter name:";
    Weapon::Name name = inputName();
    
    std::cout << "Enter damage:";
    Weapon::Damage damage = inputInt();
    
    std::cout << "Enter attackRange:";
    Weapon::AttackRange attackRange = inputInt();
    
    array[rangedWeapon] = new RangedWeapon{name, damage, attackRange};
    
    std::cout << "Do you want to change something?\n"
                 "1) - yes\n"
                 "2) - no\n"
                 ">";
    
    int choice;
    while (true) {
        choice = inputInt();
        if (choice == 1 || choice == 2) {
            break;
        } else {
            std::cout << "Enter 1 or 2" << std::endl;
        }
    }
    
    if (choice == 1) {
        std::cout << "What to change?\n"
                     "1) - name\n"
                     "2) - damage\n"
                     "3) - attackRange\n"
                     ">";
        
        while (true) {
            choice = inputInt();
            if (choice == 1 || choice == 2 || choice == 3) {
                break;
            } else {
                std::cout << "Enter 1 or 2 or 3" << std::endl;
            }
        }
        
        switch (choice) {
        case 1:
            std::cout << "Enter name:";
            name = inputName();
            array[rangedWeapon]->setName(name);
            break;
        
        case 2:
            std::cout << "Enter damage:";
            damage = inputInt();
            array[rangedWeapon]->setDamage(damage);
            break;
        
        case 3:
            std::cout << "Enter attackRange:";
            attackRange = inputInt();
            array[rangedWeapon]->setAttackRange(attackRange);
            break;
        }
    }
    //-----------------------------------------------------------------------//
    
    array[firearm]       = new Firearm{30};
    array[coldSteelArms] = new ColdSteelArms{3, 40, 300};
    
    // Polymorphism magic
    for (int i = 0; i < totalPointers; ++i) {
        std::cout << std::endl;
        array[i]->printType();
        std::cout << "Counted lethality is " << array[i]->countLethality() << std::endl;
        array[i]->display();
        
        // Deallocate memory
        delete array[i];
    }
    delete[] array;
    
    system("pause");
    return 0;
}

Weapon::Name inputName() {
    Weapon::Name name;
    
    while (true) {
        std::cin >> name;
        if (std::cin.fail() || std::cin.bad()) {
            std::cout << "Invalid input" << std::endl;
            std::cin.clear();
            std::cin.ignore(10'000, '\n');
        } else if (std::cin.peek() != '\n') {
            std::cout << "Enter only single word" << std::endl;
            std::cin.ignore(10'000, '\n');
        } else {
            bool isGood = true;
            for (auto &item: name) {
                if (!std::islower(item) && !std::isupper(item)) {
                    std::cout << "Use only letters!" << std::endl;
                    std::cin.ignore(10'000, '\n');
                    isGood = false;
                    break;
                }
            }
            if (isGood) {
                break;
            }
        }
    }
    
    return name;
}

int inputInt() {
    int number;
    
    while (true) {
        std::cin >> number;
        if (std::cin.fail() || std::cin.bad()) {
            std::cout << "Invalid input" << std::endl;
            std::cin.clear();
            std::cin.ignore(10'000, '\n');
        } else if (std::cin.peek() != '\n') {
            std::cout << "Use only digits" << std::endl;
            std::cin.ignore(10'000, '\n');
        } else if (number < 0) {
            std::cout << "Enter only positive values" << std::endl;
        } else {
            break;
        }
    }
    
    return number;
}