#ifndef LAB6_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_
#define LAB6_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_

#include "../Weapon/Weapon.hpp"

class ColdSteelArms : public Weapon {
    //--------------------------------Fields---------------------------------//
    
    // No new fields
    // Used Weapon fields:
    // killCounter
    // attackSpeed
    // price
    // lethality
    
    //----------------------Constructors & Destructors-----------------------//
public:
    ColdSteelArms(Counter killCounter, AttackSpeed attackSpeed, Price price);
    
    ~ColdSteelArms() override;
    
    //---------------------------Getters & Setters---------------------------//
public:
    using Weapon::getKillCounter;
    using Weapon::setKillCounter;
    
    using Weapon::getAttackSpeed;
    using Weapon::setAttackSpeed;
    
    using Weapon::getPrice;
    using Weapon::setPrice;
    
    using Weapon::getLethality;

private: // Delete some getters & setters
    using Weapon::getName;
    using Weapon::setName;
    
    using Weapon::getDamage;
    using Weapon::setDamage;
    
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


#endif //LAB6_SRC_COLDSTEELARMS_COLDSTEELARMS_HPP_