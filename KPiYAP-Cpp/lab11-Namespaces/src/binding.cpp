//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "binding.hpp"

std::ostream &book::operator<<(std::ostream &os, const book::Binding &binding) {
  switch (binding.type_) {
    case Binding::Type::HARD: os << "hard";
      break;

    case Binding::Type::SOFT: os << "soft";
      break;
  }
  return os;
}

book::Binding::Binding() {
}

book::Binding::Binding(book::Binding::Type type)
    : type_(type) {
}

book::Binding::~Binding() {
}

book::Binding::Type book::Binding::GetType() const {
  return type_;
}

void book::Binding::SetType(book::Binding::Type type) {
  type_ = type;
}
bool book::Binding::operator==(const book::Binding &rhs) const {
  return type_ == rhs.type_;
}

bool book::Binding::operator!=(const book::Binding &rhs) const {
  return !(rhs == *this);
}


