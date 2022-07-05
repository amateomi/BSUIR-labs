#include "Weapon.hpp"

#include <iostream>

//------------------------Constructors & Destructors-------------------------//

Weapon::Weapon(const Weapon::Name &name, Weapon::Counter killCounter, Weapon::Damage damage,
               Weapon::AttackSpeed attackSpeed, Weapon::AttackRange attackRange, Weapon::Price price) {
    std::cout << "Weapon constructor called" << std::endl;
    setName(name);
    setKillCounter(killCounter);
    setDamage(damage);
    setAttackSpeed(attackSpeed);
    setAttackRange(attackRange);
    setPrice(price);
    setLethality();
}

Weapon::~Weapon() {
    std::cout << "Weapon destructor called" << std::endl;
}

//-----------------------------Getters & Setters-----------------------------//

const Weapon::Name &Weapon::getName() const {
    return name;
}

void Weapon::setName(const Weapon::Name &newName) {
    Weapon::name = newName;
}

Weapon::Counter Weapon::getKillCounter() const {
    return killCounter;
}

void Weapon::setKillCounter(Weapon::Counter newKillCounter) {
    Weapon::killCounter = newKillCounter;
}

Weapon::Damage Weapon::getDamage() const {
    return damage;
}

void Weapon::setDamage(Weapon::Damage newDamage) {
    Weapon::damage = newDamage;
}

Weapon::AttackSpeed Weapon::getAttackSpeed() const {
    return attackSpeed;
}

void Weapon::setAttackSpeed(Weapon::AttackSpeed newAttackSpeed) {
    Weapon::attackSpeed = newAttackSpeed;
}

Weapon::AttackRange Weapon::getAttackRange() const {
    return attackRange;
}

void Weapon::setAttackRange(Weapon::AttackRange newAttackRange) {
    Weapon::attackRange = newAttackRange;
}

Weapon::Price Weapon::getPrice() const {
    return price;
}

void Weapon::setPrice(Weapon::Price newPrice) {
    Weapon::price = newPrice;
}

Weapon::Lethality Weapon::getLethality() const {
    return lethality;
}

void Weapon::setLethality(Weapon::Lethality newLethality) {
    lethality = newLethality;
}