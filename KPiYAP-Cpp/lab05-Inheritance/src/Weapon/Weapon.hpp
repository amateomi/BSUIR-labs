#ifndef LAB5_SRC_WEAPON_WEAPON_HPP_
#define LAB5_SRC_WEAPON_WEAPON_HPP_

#include <iostream>

class Weapon {
    //------------------------------User Types-------------------------------//
public:
    using Counter = int;

    //--------------------------------Fields---------------------------------//
private:
    Counter killCounter{};

    //----------------------Constructors & Destructors-----------------------//
public:
    explicit Weapon(Counter killCounter);

    virtual ~Weapon();

    //---------------------------Getters & Setters---------------------------//
public:
    Counter getKillCounter() const;

    void setKillCounter(Counter newKillCounter);

    //--------------------------------Output---------------------------------//
public:
    virtual void show() const;
};

#endif //LAB5_SRC_WEAPON_WEAPON_HPP_
