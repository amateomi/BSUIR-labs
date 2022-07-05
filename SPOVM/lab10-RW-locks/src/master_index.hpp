#pragma once

#include <unordered_map>

#include "rwlock_manager.hpp"
#include "file_manager.hpp"

namespace prod {

class MasterIndex : public RwlockManager, public FileManager {
 public:
  explicit MasterIndex(const fs::path &file_path);

  void addRecord(int id, std::ios::pos_type record_pos);
  void delRecord(int id);

  // Return -1 when id is not in table
  std::ios::pos_type getPosition(int id) const;

  void updateFile() override;

 private:
  std::unordered_map<int, std::ios::pos_type> map_;
};

} // prod
