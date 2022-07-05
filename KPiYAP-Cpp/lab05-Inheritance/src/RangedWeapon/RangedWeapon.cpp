#include "RangedWeapon.hpp"

//------------------------Constructors & Destructors-------------------------//

RangedWeapon::RangedWeapon(Weapon::Counter killCounter, RangedWeapon::FiringRange firingRange)
        : Weapon(killCounter) {
    std::cout << "RangedWeapon constructor called\n";
    setFiringRange(firingRange);
}

RangedWeapon::~RangedWeapon() {
    std::cout << "RangedWeapon destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

RangedWeapon::FiringRange RangedWeapon::getFiringRange() const {
    return firingRange;
}

void RangedWeapon::setFiringRange(RangedWeapon::FiringRange newFiringRange) {
    RangedWeapon::firingRange = newFiringRange;
}

//----------------------------------Output-----------------------------------//

void RangedWeapon::show() const {
    Weapon::show();
    std::cout << "Firing range: " << firingRange << std::endl;
}
