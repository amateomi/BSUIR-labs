#include "file_manager.hpp"

FileManager::FileManager(const fs::path &file_path)
    : FILE_PATH(file_path) {
  file_.open(FILE_PATH, std::ios::in | std::ios::out | std::ios::ate);
  if (!file_.is_open()) {
    throw std::domain_error("Failed to open " + file_path.string());
  }
  file_.seekg(0, std::ios::beg);
}

FileManager::~FileManager() {
  file_.close();
}
