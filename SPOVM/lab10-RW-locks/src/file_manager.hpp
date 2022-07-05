#pragma once

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class FileManager {
 public:
  explicit FileManager(const fs::path &file_path);
  ~FileManager();

  virtual void updateFile() = 0;

  const fs::path FILE_PATH;

 protected:
  std::fstream file_;
};
