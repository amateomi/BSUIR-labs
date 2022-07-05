//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_AUTHOR_HPP_
#define LAB11_SRC_AUTHOR_HPP_

#include <iostream>
#include <string>
#include <utility>

namespace book {

class Author {
 public:
  // Constructors
  Author();
  explicit Author(std::string full_name);

  // Destructor
  virtual ~Author();

  // Getters and setters
  [[nodiscard]] const std::string &GetFullName() const;
  void SetFullName(const std::string &full_name);

  bool operator==(const Author &rhs) const;
  bool operator!=(const Author &rhs) const;

 private:
  std::string full_name_;
};

}

#endif //LAB11_SRC_AUTHOR_HPP_
