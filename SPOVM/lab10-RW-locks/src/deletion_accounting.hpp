#pragma once

#include <list>

#include "rwlock_manager.hpp"
#include "file_manager.hpp"

namespace prod {

class DeletionAccounting : public RwlockManager, public FileManager {
 public:
  explicit DeletionAccounting(const fs::path &file_path);

  void addDeletion(std::ios::pos_type del_pos);

  // Return -1 when no empty positions
  std::ios::pos_type getEmptyPosition();

  void updateFile() override;

 private:
  std::list<std::ios::pos_type> list_;
};

} // prod
