#ifndef LAB6_SRC_WEAPON_WEAPON_HPP_
#define LAB6_SRC_WEAPON_WEAPON_HPP_

#include <string>

class Weapon {
    //------------------------------User Types-------------------------------//
public:
    using Name = std::string;
    using Counter = int;
    using Damage = int;
    using AttackSpeed = double;
    using AttackRange = double;
    using Price = double;
    using Lethality = double;
    
    //--------------------------------Fields---------------------------------//
private:
    Name        name{};
    Counter     killCounter{};
    Damage      damage{};
    AttackSpeed attackSpeed{};
    AttackRange attackRange{};
    Price       price{};
    Lethality   lethality{}; // To calculate this value call countLethality() method
    
    //----------------------Constructors & Destructors-----------------------//
public:
    // Set lethality to -1.0
    Weapon(const Name &name, Counter killCounter, Damage damage,
           AttackSpeed attackSpeed, AttackRange attackRange, Price price);
    
    virtual ~Weapon();
    
    //---------------------------Getters & Setters---------------------------//
public:
    const Name &getName() const;
    
    void setName(const Name &newName);
    
    Counter getKillCounter() const;
    
    void setKillCounter(Counter newKillCounter);
    
    Damage getDamage() const;
    
    void setDamage(Damage damage);
    
    AttackSpeed getAttackSpeed() const;
    
    void setAttackSpeed(AttackSpeed attackSpeed);
    
    AttackRange getAttackRange() const;
    
    virtual void setAttackRange(AttackRange attackRange) final;
    
    Price getPrice() const;
    
    void setPrice(Price price);
    
    Lethality getLethality() const;

protected:
    // Used in Weapon constructor for beauty,
    // also used in override countLethality() method
    // By default set lethality to -1.0
    void setLethality(Lethality newLethality = -1.0);
    
    //---------------------------------Print---------------------------------//
public:
    // Display all fields
    virtual void display() const = 0;
    
    // Display class name
    virtual void printType() const = 0;
    
    //---------------------------------Count---------------------------------//
public:
    // Count specific lethality coefficient for heirs
    virtual Lethality countLethality() = 0;
};

#endif //LAB6_SRC_WEAPON_WEAPON_HPP_
