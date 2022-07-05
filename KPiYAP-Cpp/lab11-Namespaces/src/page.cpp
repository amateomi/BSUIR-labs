//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "page.hpp"

book::Page::Page() {
}

book::Page::Page(int number, std::string text)
    : number_(number), text_(std::move(text)) {
}

book::Page::~Page() {
}

int book::Page::GetNumber() const {
  return number_;
}

void book::Page::SetNumber(int number) {
  number_ = number;
}

const std::string &book::Page::GetText() const {
  return text_;
}

void book::Page::SetText(const std::string &text) {
  text_ = text;
}
bool book::Page::operator==(const book::Page &rhs) const {
  return number_ == rhs.number_ && text_ == rhs.text_;
}

bool book::Page::operator!=(const book::Page &rhs) const {
  return !(rhs == *this);
}

