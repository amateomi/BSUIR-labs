#ifndef LAB5_SRC_FIREARM_FIREARM_HPP_
#define LAB5_SRC_FIREARM_FIREARM_HPP_

#include "../RangedWeapon/RangedWeapon.hpp"

class Firearm : public RangedWeapon {
    //------------------------------User Types-------------------------------//
public:
    using Caliber = double;

    //--------------------------------Fields---------------------------------//
private:
    Caliber caliber{};

    //----------------------Constructors & Destructors-----------------------//
public:
    Firearm(Counter killCounter, FiringRange firingRange, Caliber caliber);

    ~Firearm() override;

    //---------------------------Getters & Setters---------------------------//
public:
    Caliber getCaliber() const;

    void setCaliber(Caliber newCaliber);

    //--------------------------------Output---------------------------------//
public:
    void show() const override;
};

#endif //LAB5_SRC_FIREARM_FIREARM_HPP_
