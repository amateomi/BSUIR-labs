//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "publisher.hpp"

book::Publisher::Publisher() {
}

book::Publisher::Publisher(std::string title)
    : title_(std::move(title)) {
}

book::Publisher::~Publisher() {
}

const std::string &book::Publisher::GetTitle() const {
  return title_;
}

void book::Publisher::SetTitle(const std::string &title) {
  title_ = title;
}
bool book::Publisher::operator==(const book::Publisher &rhs) const {
  return title_ == rhs.title_;
}

bool book::Publisher::operator!=(const book::Publisher &rhs) const {
  return !(rhs == *this);
}
