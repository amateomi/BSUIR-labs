#include "ColdSteelArms.hpp"

//------------------------Constructors & Destructors-------------------------//
ColdSteelArms::ColdSteelArms(Weapon::Counter killCounter, MeleeWeapon::StrikeRange strikeRange,
                             const ColdSteelArms::BladeMaterial& bladeMaterial)
        : Weapon(killCounter), MeleeWeapon(killCounter, strikeRange) {
    std::cout << "ColdSteelArms constructor called\n";
    setBladeMaterial(bladeMaterial);
}

ColdSteelArms::~ColdSteelArms() {
    std::cout << "ColdSteelArms destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

const ColdSteelArms::BladeMaterial &ColdSteelArms::getBladeMaterial() const {
    return bladeMaterial;
}

void ColdSteelArms::setBladeMaterial(const ColdSteelArms::BladeMaterial &newBladeMaterial) {
    ColdSteelArms::bladeMaterial = newBladeMaterial;
}

//----------------------------------Output-----------------------------------//

void ColdSteelArms::show() const {
    MeleeWeapon::show();
    std::cout << "Blade material: " << bladeMaterial << std::endl;
}
