#include "MosinRifle.hpp"

//------------------------Constructors & Destructors-------------------------//

MosinRifle::MosinRifle(Weapon::Counter killCounter,
                       RangedWeapon::FiringRange firingRange, Firearm::Caliber caliber,
                       MeleeWeapon::StrikeRange strikeRange, const ColdSteelArms::BladeMaterial &bladeMaterial,
                       MosinRifle::ModificationType modificationType)
        : Weapon(killCounter),
          Firearm(killCounter, firingRange, caliber),
          ColdSteelArms(killCounter, strikeRange, bladeMaterial) {
    std::cout << "MosinRifle constructor called\n";
    setModificationType(modificationType);
}

MosinRifle::~MosinRifle() {
    std::cout << "MosinRifle destructor called\n";
}

//-----------------------------Getters & Setters-----------------------------//

MosinRifle::ModificationType MosinRifle::getModificationType() const {
    return modificationType;
}

std::string MosinRifle::getModificationTypeToPrint() const {
    std::string string;

    switch (modificationType) {
    case ModificationType::Sport:
        string = "sport";
        break;

    case ModificationType::Hunt:
        string = "hunt";
        break;

    case ModificationType::Combat:
        string = "combat";
        break;
    }

    return string;
}

void MosinRifle::setModificationType(MosinRifle::ModificationType newModificationType) {
    modificationType = newModificationType;
}

//----------------------------------Output-----------------------------------//

void MosinRifle::show() const {
    std::cout << "Kill counter: " << getKillCounter() << std::endl;
    std::cout << "Firing range: " << getFiringRange() << std::endl;
    std::cout << "Caliber: " << getCaliber() << std::endl;
    std::cout << "Strike range: " << getStrikeRange() << std::endl;
    std::cout << "Blade material: " << getBladeMaterial() << std::endl;
    std::cout << "Modification: " << getModificationTypeToPrint() << std::endl;
}
