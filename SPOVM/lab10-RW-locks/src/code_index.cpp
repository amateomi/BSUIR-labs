#include "code_index.hpp"

namespace prod {

CodeIndex::CodeIndex(const fs::path &index_path)
    : RwlockManager("/code_index"),
      FileManager(index_path) {
  updateFile();
}

void CodeIndex::addRecord(int code, int id) {
  if (!isAvailableCode(code)) {
    throw std::runtime_error("Attempt to insert not unique code");
  }

  writeLock();
  try {
    file_.seekp(0, std::ios::end);
    file_ << code << ' ' << id << '\n';
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

void CodeIndex::delRecord(int code) {
  writeLock();
  try {
    map_.erase(code);

    file_.close();
    file_.open(FILE_PATH, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open code index during deletion");
    }

    for (const auto &[rec_code, rec_id] : map_) {
      file_ << rec_code << ' ' << rec_id << '\n';
    }
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

int CodeIndex::getId(int code) const {
  if (auto id = map_.find(code); id == map_.end()) {
    throw std::runtime_error("No such code in code index");
  } else {
    return id->second;
  }
}

bool CodeIndex::isAvailableCode(int code) const {
  return map_.find(code) == map_.end();
}

void CodeIndex::updateFile() {
  readLock();
  try {
    map_.clear();

    file_.seekg(0, std::ios::beg);
    while (true) {
      int code;
      file_ >> code;
      if (file_.eof()) {
        file_.clear();
        break;
      }
      if (file_.fail() || code < 0) {
        throw std::domain_error("Invalid code information");
      }

      int id;
      file_ >> id;
      if (file_.fail() || id < 0) {
        throw std::domain_error("Invalid id information");
      }

      map_[code] = id;
    }
  } catch (std::exception &exception) {
    unlock();
    throw std::domain_error(exception.what());
  }
  unlock();
}

} // prod
