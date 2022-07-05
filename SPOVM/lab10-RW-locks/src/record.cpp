#include "record.hpp"

#include <iomanip>

namespace prod {

std::istream &operator>>(std::istream &is, Record &record) {
  is >> record.name;
  if (record.name.empty() || record.name.length() > 16) {
    throw std::runtime_error("Invalid name length");
  }
  is >> record.code;
  if (record.code < 0) {
    throw std::runtime_error("Invalid code (negative value)");
  }
  is >> record.amount;
  if (record.amount < 0) {
    throw std::runtime_error("Invalid amount (negative value)");
  }
  is >> record.reserved;
  if (record.reserved < 0) {
    throw std::runtime_error("Invalid reserved (negative value)");
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const Record &record) {
  os << std::setw(10) << std::left << record.id << " "
     << std::setw(16) << std::left << record.name << " "
     << std::setw(10) << std::left << record.code << " "
     << std::setw(10) << std::left << record.amount << " "
     << std::setw(10) << std::left << record.reserved << "\n";
  return os;
}

} // prod
