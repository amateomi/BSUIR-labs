#ifndef LAB5_SRC_MOSINRIFLE_MOSINRIFLE_HPP_
#define LAB5_SRC_MOSINRIFLE_MOSINRIFLE_HPP_

#include "../Firearm/Firearm.hpp"
#include "../ColdSteelArms/ColdSteelArms.hpp"

class MosinRifle : public Firearm, public ColdSteelArms {
    //------------------------------User Types-------------------------------//
public:
    enum class ModificationType {
        Sport,
        Hunt,
        Combat
    };

    //--------------------------------Fields---------------------------------//
private:
    ModificationType modificationType{};

    //----------------------Constructors & Destructors-----------------------//
public:
    MosinRifle(Counter killCounter,
               FiringRange firingRange, Caliber caliber,
               StrikeRange strikeRange, const BladeMaterial &bladeMaterial,
               ModificationType modificationType);

    ~MosinRifle() override;
    //---------------------------Getters & Setters---------------------------//
public:
    ModificationType getModificationType() const;

    std::string getModificationTypeToPrint() const;

    void setModificationType(ModificationType newModificationType);

    //--------------------------------Output---------------------------------//
public:
    void show() const override;
};

#endif //LAB5_SRC_MOSINRIFLE_MOSINRIFLE_HPP_
