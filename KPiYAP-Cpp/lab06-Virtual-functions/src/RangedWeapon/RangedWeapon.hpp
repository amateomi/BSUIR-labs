#ifndef LAB6_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_
#define LAB6_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_

#include "../Weapon/Weapon.hpp"

class RangedWeapon : public Weapon {
    //--------------------------------Fields---------------------------------//
    
    // No new fields
    // Used Weapon fields:
    // name
    // damage
    // attackRange
    // lethality
    
    //----------------------Constructors & Destructors-----------------------//
public:
    RangedWeapon(const Name &name, Damage damage, AttackRange attackRange);
    
    ~RangedWeapon() override;
    
    //---------------------------Getters & Setters---------------------------//
public:
    using Weapon::getName;
    using Weapon::setName;
    
    using Weapon::getDamage;
    using Weapon::setDamage;
    
    using Weapon::getAttackRange;
    using Weapon::setAttackRange;
    
    using Weapon::getLethality;

private: // Delete some getters & setters
    using Weapon::getKillCounter;
    using Weapon::setKillCounter;
    
    using Weapon::getAttackSpeed;
    using Weapon::setAttackSpeed;
    
    using Weapon::getPrice;
    using Weapon::setPrice;
    
    //---------------------------------Print---------------------------------//
public:
    void display() const override;
    
    void printType() const override;
    
    //---------------------------------Count---------------------------------//
public:
    Lethality countLethality() override;
};

#endif //LAB6_SRC_RANGEDWEAPON_RANGEDWEAPON_HPP_
