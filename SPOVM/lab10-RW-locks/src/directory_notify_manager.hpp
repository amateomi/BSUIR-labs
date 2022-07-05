#pragma once

#include <filesystem>

namespace fs = std::filesystem;

class DirectoryNotifyManager {
 public:
  DirectoryNotifyManager(const fs::path &dir, void (* handler)(int));
  ~DirectoryNotifyManager();

  // Used after each SIGIO in notify_handler form table.cpp
  void resetNotify(void (* handler)(int)) const;

 private:
  int fd{};
};
