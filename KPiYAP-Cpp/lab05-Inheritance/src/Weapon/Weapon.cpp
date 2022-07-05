#include "Weapon.hpp"

//------------------------Constructors & Destructors-------------------------//

Weapon::Weapon(Weapon::Counter killCounter) {
    std::cout << "Weapon constructor called\n";
    setKillCounter(killCounter);
}

Weapon::~Weapon() {
    std::cout << "Weapon destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

Weapon::Counter Weapon::getKillCounter() const {
    return killCounter;
}

void Weapon::setKillCounter(Weapon::Counter newKillCounter) {
    this->killCounter = newKillCounter;
}

//----------------------------------Output-----------------------------------//

void Weapon::show() const {
    std::cout << "KillCounter: " << killCounter << std::endl;
}
