//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_BINDING_HPP_
#define LAB11_SRC_BINDING_HPP_

#include <iostream>

namespace book {

class Binding {
 public:
  // Book binding type
  enum class Type {
    HARD,
    SOFT
  };

  friend std::ostream &operator<<(std::ostream &os, const Binding &binding);

  // Constructors
  Binding();
  explicit Binding(Type type);

  // Destructor
  virtual ~Binding();

  // Getters and setters
  [[nodiscard]] Type GetType() const;
  void SetType(Type type);

  bool operator==(const Binding &rhs) const;
  bool operator!=(const Binding &rhs) const;

 private:
  Type type_{};
};

}

#endif //LAB11_SRC_BINDING_HPP_
