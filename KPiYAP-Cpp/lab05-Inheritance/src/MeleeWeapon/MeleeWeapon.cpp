#include "MeleeWeapon.hpp"

//------------------------Constructors & Destructors-------------------------//

MeleeWeapon::MeleeWeapon(MeleeWeapon::Counter killCounter, MeleeWeapon::StrikeRange strikeRange)
        : Weapon(killCounter) {
    std::cout << "MeleeWeapon constructor called\n";
    setStrikeRange(strikeRange);
}

MeleeWeapon::~MeleeWeapon() {
    std::cout << "MeleeWeapon destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

MeleeWeapon::StrikeRange MeleeWeapon::getStrikeRange() const {
    return strikeRange;
}

void MeleeWeapon::setStrikeRange(MeleeWeapon::StrikeRange newStrikeRange) {
    MeleeWeapon::strikeRange = newStrikeRange;
}

//----------------------------------Output-----------------------------------//

void MeleeWeapon::show() const {
    Weapon::show();
    std::cout << "Strike range: " << strikeRange << std::endl;
}
