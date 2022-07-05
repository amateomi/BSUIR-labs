#pragma once

#include <string>

namespace prod {

struct Record {
  constexpr static std::string_view INPUT_FORMAT{
      "<NAME> <CODE> <AMOUNT> <RESERVED>\n"
      "(NAME max len is 16 characters, others are positive integer values)"
  };
  constexpr static std::string_view TABLE_FORMAT{
      "ID         NAME             CODE       AMOUNT     RESERVED"
  };

  friend std::istream &operator>>(std::istream &is, Record &record);
  friend std::ostream &operator<<(std::ostream &os, const Record &record);

  int         id{};
  std::string name;
  int         code{};
  int         amount{};
  int         reserved{};
};

} // prod
