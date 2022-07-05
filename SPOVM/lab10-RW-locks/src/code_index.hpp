#pragma once

#include <map>

#include "rwlock_manager.hpp"
#include "file_manager.hpp"

namespace prod {

class CodeIndex : public RwlockManager, public FileManager {
 public:
  explicit CodeIndex(const fs::path &file_path);

  void addRecord(int code, int id);
  void delRecord(int code);

  int getId(int code) const;
  bool isAvailableCode(int code) const;

  void updateFile() override;

 private:
  std::map<int, int> map_;
};

} // prod
