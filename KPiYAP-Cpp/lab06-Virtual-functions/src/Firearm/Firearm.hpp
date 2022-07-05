#ifndef LAB6_SRC_FIREARM_FIREARM_HPP_
#define LAB6_SRC_FIREARM_FIREARM_HPP_

#include "../RangedWeapon/RangedWeapon.hpp"

class Firearm : public RangedWeapon {
    //--------------------------------Fields---------------------------------//
    
    // No new fields
    // Used Weapon fields:
    // damage
    // lethality
    
    //----------------------Constructors & Destructors-----------------------//
public:
    explicit Firearm(Damage damage);
    
    ~Firearm() override;
    
    //---------------------------Getters & Setters---------------------------//
public:
    using Weapon::getDamage;
    using Weapon::setDamage;
    
    using Weapon::getLethality;

private: // Delete some getters & setters
    
    using Weapon::getName;
    using Weapon::setName;
    
    using Weapon::getAttackRange;
    using Weapon::setAttackRange;
    
    //---------------------------------Print---------------------------------//
public:
    void display() const override;
    
    void printType() const override;
    
    //---------------------------------Count---------------------------------//
public:
    Lethality countLethality() override;
};

#endif //LAB6_SRC_FIREARM_FIREARM_HPP_