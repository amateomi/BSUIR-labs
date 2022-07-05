#ifndef LAB5_SRC_MELEEWEAPON_MELEEWEAPON_HPP_
#define LAB5_SRC_MELEEWEAPON_MELEEWEAPON_HPP_

#include "../Weapon/Weapon.hpp"

class MeleeWeapon : public virtual Weapon {
    //------------------------------User Types-------------------------------//
public:
    using StrikeRange = double;

    //--------------------------------Fields---------------------------------//
private:
    StrikeRange strikeRange{};

    //----------------------Constructors & Destructors-----------------------//
public:
    MeleeWeapon(Counter killCounter, StrikeRange strikeRange);

    ~MeleeWeapon() override;
    //---------------------------Getters & Setters---------------------------//
public:
    StrikeRange getStrikeRange() const;

    void setStrikeRange(StrikeRange newStrikeRange);

    //--------------------------------Output---------------------------------//
public:
    void show() const override;
};


#endif //LAB5_SRC_MELEEWEAPON_MELEEWEAPON_HPP_
