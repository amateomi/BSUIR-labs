#include "deletion_accounting.hpp"

namespace prod {

DeletionAccounting::DeletionAccounting(const fs::path &file_path)
    : RwlockManager("/deletion_accounting"),
      FileManager(file_path) {
  updateFile();
}

void DeletionAccounting::addDeletion(std::ios::pos_type del_pos) {
  writeLock();
  try {
    file_.seekp(0, std::ios::end);
    file_ << del_pos << '\n';
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

std::ios::pos_type DeletionAccounting::getEmptyPosition() {
  if (list_.empty()) {
    return -1;
  }

  std::ios::pos_type pos;
  writeLock();
  try {
    pos = list_.front();
    list_.pop_front();

    file_.close();
    file_.open(FILE_PATH, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open deletion accounting in getEmptyPosition");
    }

    for (const auto &empty_pos : list_) {
      file_ << empty_pos << '\n';
    }
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
  return pos;
}

void DeletionAccounting::updateFile() {
  readLock();
  try {
    list_.clear();

    file_.seekg(0, std::ios::beg);
    while (true) {
      std::streamoff empty_pos;
      file_ >> empty_pos;
      if (file_.eof()) {
        file_.clear();
        break;
      }
      if (file_.fail() || empty_pos < 0) {
        throw std::domain_error("Invalid empty position information");
      }

      list_.emplace_front(empty_pos);
    }
  } catch (std::exception &exception) {
    unlock();
    throw std::domain_error(exception.what());
  }
  unlock();
}

} // prod
