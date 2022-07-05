#ifndef LAB5_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_
#define LAB5_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_

#include "../Weapon/Weapon.hpp"

class RangedWeapon : public virtual Weapon {
    //------------------------------User Types-------------------------------//
public:
    using FiringRange = double;

    //--------------------------------Fields---------------------------------//
private:
    FiringRange firingRange{};

    //----------------------Constructors & Destructors-----------------------//
public:
    RangedWeapon(Counter killCounter, FiringRange firingRange);

    ~RangedWeapon() override;

    //---------------------------Getters & Setters---------------------------//
public:
    FiringRange getFiringRange() const;

    void setFiringRange(FiringRange newFiringRange);

    //--------------------------------Output---------------------------------//
public:
    void show() const override;
};


#endif //LAB5_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_
