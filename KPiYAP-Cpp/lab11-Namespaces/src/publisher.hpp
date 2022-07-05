//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_PUBLISHER_HPP_
#define LAB11_SRC_PUBLISHER_HPP_

#include <iostream>
#include <string>
#include <utility>

namespace book {

class Publisher {
 public:
  // Constructors
  Publisher();
  explicit Publisher(std::string title);

  // Destructor
  virtual ~Publisher();

  // Getters and setters
  [[nodiscard]] const std::string &GetTitle() const;
  void SetTitle(const std::string &title);

  bool operator==(const Publisher &rhs) const;
  bool operator!=(const Publisher &rhs) const;

 private:
  std::string title_;
};

}

#endif //LAB11_SRC_PUBLISHER_HPP_
