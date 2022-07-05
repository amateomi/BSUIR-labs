#include "Firearm.hpp"

//------------------------Constructors & Destructors-------------------------//

Firearm::Firearm(Weapon::Counter killCounter, RangedWeapon::FiringRange firingRange, Firearm::Caliber caliber)
        : Weapon(killCounter), RangedWeapon(killCounter, firingRange) {
    std::cout << "Firearm constructor called\n";
    setCaliber(caliber);
}

Firearm::~Firearm() {
    std::cout << "Firearm destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

Firearm::Caliber Firearm::getCaliber() const {
    return caliber;
}

void Firearm::setCaliber(Firearm::Caliber newCaliber) {
    Firearm::caliber = newCaliber;
}

//----------------------------------Output-----------------------------------//

void Firearm::show() const {
    RangedWeapon::show();
    std::cout << "Caliber: " << caliber << std::endl;
}
