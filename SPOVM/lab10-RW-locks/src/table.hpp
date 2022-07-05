#pragma once

#include <variant>

#include "directory_notify_manager.hpp"
#include "record.hpp"
#include "deletion_accounting.hpp"
#include "master_index.hpp"
#include "name_index.hpp"
#include "code_index.hpp"

namespace prod {

class Table : public RwlockManager, public FileManager, public DirectoryNotifyManager {
 public:
  explicit Table(const fs::path &file_path);

  void addRecord(Record &record);
  void delRecord(int id);
  Record getRecord(int id);
  void putRecord(const Record &new_record);
  std::variant<std::list<int>, int> getPrimary(std::variant<std::string_view, int> index) const;

  void updateFile() override;

 private:
  std::ios::pos_type seed_file_pos_{};
  std::ios::pos_type amount_file_pos_{};

  prod::DeletionAccounting deletion_accounting_;
  prod::MasterIndex        master_index_;
  prod::NameIndex          name_index_;
  prod::CodeIndex          code_index_;
};

} // prod
