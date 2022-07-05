#pragma once

#include "table.hpp"

class App {
 public:
  App(const std::filesystem::path &table_path);

  void run();

 private:
  void addRecord();
  void delRecord();
  void getRecord();
  void putRecord();
  void getPrimary() const;

  prod::Table table_;
};
