//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_PAGE_HPP_
#define LAB11_SRC_PAGE_HPP_

#include <iostream>
#include <string>
#include <utility>

namespace book {

class Page {
 public:
  // Constructors
  Page();
  Page(int number, std::string text);

  // Destructor
  virtual ~Page();

  // Getter and setters
  [[nodiscard]] int GetNumber() const;
  void SetNumber(int number);

  [[nodiscard]] const std::string &GetText() const;
  void SetText(const std::string &text);

  bool operator==(const Page &rhs) const;
  bool operator!=(const Page &rhs) const;

 private:
  int number_{};
  std::string text_;
};

}

#endif //LAB11_SRC_PAGE_HPP_
