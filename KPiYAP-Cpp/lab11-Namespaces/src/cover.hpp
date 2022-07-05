//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_COVER_HPP_
#define LAB11_SRC_COVER_HPP_

#include <iostream>
#include <string>
#include <utility>

namespace book {

class Cover {
 public:
  // Constructors
  Cover();
  explicit Cover(std::string title);

  // Destructor
  virtual ~Cover();

  // Getters and setters
  [[nodiscard]] const std::string &GetTitle() const;
  void SetTitle(const std::string &title);

  bool operator==(const Cover &rhs) const;
  bool operator!=(const Cover &rhs) const;

 private:
  std::string title_;
};

}

#endif //LAB11_SRC_COVER_HPP_
