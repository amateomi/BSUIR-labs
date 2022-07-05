#pragma once

#include <list>
#include <map>

#include "rwlock_manager.hpp"
#include "file_manager.hpp"

namespace prod {

class NameIndex : public RwlockManager, public FileManager {
 public:
  explicit NameIndex(const fs::path &file_path);

  void addRecord(std::string_view name, int id);
  void delRecord(std::string_view name);

  // Return empty list when code is not in index file
  std::list<int> getId(std::string_view name) const;

  void updateFile() override;

 private:
  std::multimap<std::string, int> map_;
};

} // prod
