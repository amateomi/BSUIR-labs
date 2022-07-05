#include "ColdSteelArms.hpp"

#include <iostream>

//------------------------Constructors & Destructors-------------------------//

ColdSteelArms::ColdSteelArms(Weapon::Counter killCounter, Weapon::AttackSpeed attackSpeed, Weapon::Price price)
        : Weapon{"\anone", killCounter, -1, attackSpeed, -1.0, price} {
    std::cout << "ColdSteelArms constructor called" << std::endl;
}

ColdSteelArms::~ColdSteelArms() {
    std::cout << "ColdSteelArms destructor called" << std::endl;
}

//-----------------------------------Print-----------------------------------//

void ColdSteelArms::display() const {
    std::cout << std::endl
              << "KillCounter: " << getKillCounter() << std::endl
              << "AttackSpeed: " << getAttackSpeed() << std::endl
              << "Price: " << getPrice() << std::endl
              << "Lethality: " << getLethality() << std::endl
              << std::endl;
}

void ColdSteelArms::printType() const {
    std::cout << "ColdSteelArms" << std::endl;
}

//-----------------------------------Count-----------------------------------//

Weapon::Lethality ColdSteelArms::countLethality() {
    Lethality newLethality = getKillCounter() + getAttackSpeed() * 10 / 100.0;
    setLethality(newLethality);
    return newLethality;
}
