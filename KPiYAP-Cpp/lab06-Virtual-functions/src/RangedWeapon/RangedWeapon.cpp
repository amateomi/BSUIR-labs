#include "RangedWeapon.hpp"

#include <iostream>

//------------------------Constructors & Destructors-------------------------//

RangedWeapon::RangedWeapon(const Weapon::Name &name, Weapon::Damage damage, Weapon::AttackRange attackRange)
        : Weapon{name, -1, damage, -1.0, attackRange, -1.0} {
    std::cout << "RangedWeapon constructor called" << std::endl;
}

RangedWeapon::~RangedWeapon() {
    std::cout << "RangedWeapon destructor called" << std::endl;
}

//-----------------------------------Print-----------------------------------//

void RangedWeapon::display() const {
    std::cout << std::endl
              << "Name: " << getName() << std::endl
              << "Damage: " << getDamage() << std::endl
              << "Attack range: " << getAttackRange() << std::endl
              << "Lethality: " << getLethality() << std::endl
              << std::endl;
}

void RangedWeapon::printType() const {
    std::cout << "RangedWeapon" << std::endl;
}

//-----------------------------------Count-----------------------------------//

Weapon::Lethality RangedWeapon::countLethality() {
    Lethality newLethality = (getDamage() * 10.0 + getAttackRange()) / 100.0;
    setLethality(newLethality);
    return newLethality;
}
