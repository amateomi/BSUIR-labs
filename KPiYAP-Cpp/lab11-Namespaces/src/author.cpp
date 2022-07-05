//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "author.hpp"

book::Author::Author() {
}

book::Author::Author(std::string full_name)
    : full_name_(std::move(full_name)) {
}

book::Author::~Author() {
}

const std::string &book::Author::GetFullName() const {
  return full_name_;
}

void book::Author::SetFullName(const std::string &full_name) {
  full_name_ = full_name;
}
bool book::Author::operator==(const book::Author &rhs) const {
  return full_name_ == rhs.full_name_;
}

bool book::Author::operator!=(const book::Author &rhs) const {
  return !(rhs == *this);
}
