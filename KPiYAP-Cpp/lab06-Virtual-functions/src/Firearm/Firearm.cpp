#include "Firearm.hpp"

#include <iostream>

//------------------------Constructors & Destructors-------------------------//

Firearm::Firearm(Weapon::Damage damage)
        : RangedWeapon{"\anone", damage, -1.0} {
    std::cout << "Firearm constructor called" << std::endl;
}

Firearm::~Firearm() {
    std::cout << "Firearm destructor called" << std::endl;
}

//-----------------------------------Print-----------------------------------//

void Firearm::display() const {
    std::cout << std::endl
              << "Damage: " << getDamage() << std::endl
              << "Lethality: " << getLethality() << std::endl
              << std::endl;
}

void Firearm::printType() const {
    std::cout << "Firearm" << std::endl;
}

//-----------------------------------Count-----------------------------------//

Weapon::Lethality Firearm::countLethality() {
    Lethality newLethality = getDamage() / 10.0;
    setLethality(newLethality);
    return newLethality;
}
