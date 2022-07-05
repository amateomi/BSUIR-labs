//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "cover.hpp"

book::Cover::Cover() {
}

book::Cover::Cover(std::string title) : title_(std::move(title)) {
}

book::Cover::~Cover() {
}

const std::string &book::Cover::GetTitle() const {
  return title_;
}

void book::Cover::SetTitle(const std::string &title) {
  Cover::title_ = title;
}
bool book::Cover::operator==(const book::Cover &rhs) const {
  return title_ == rhs.title_;
}

bool book::Cover::operator!=(const book::Cover &rhs) const {
  return !(rhs == *this);
}

