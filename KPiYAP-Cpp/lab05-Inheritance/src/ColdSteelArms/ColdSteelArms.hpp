#ifndef LAB5_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_
#define LAB5_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_

#include <string>

#include "../MeleeWeapon/MeleeWeapon.hpp"

class ColdSteelArms : public MeleeWeapon {
    //------------------------------User Types-------------------------------//
public:
    using BladeMaterial = std::string;

    //--------------------------------Fields---------------------------------//
private:
    BladeMaterial bladeMaterial;

    //----------------------Constructors & Destructors-----------------------//
public:
    ColdSteelArms(Counter killCounter, StrikeRange strikeRange, const BladeMaterial& bladeMaterial);

    ~ColdSteelArms() override;

    //---------------------------Getters & Setters---------------------------//
public:
    const BladeMaterial &getBladeMaterial() const;

    void setBladeMaterial(const BladeMaterial &newBladeMaterial);

    //--------------------------------Output---------------------------------//
public:
    void show() const override;
};

#endif //LAB5_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_
