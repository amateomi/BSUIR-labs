#include "master_index.hpp"

namespace prod {

MasterIndex::MasterIndex(const fs::path &file_path)
    : RwlockManager("/master_index"),
      FileManager(file_path) {
  updateFile();
}

void MasterIndex::addRecord(int id, std::ios::pos_type record_pos) {
  writeLock();
  try {
    file_.seekp(0, std::ios::end);
    file_ << id << ' ' << record_pos << '\n';
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

void MasterIndex::delRecord(int id) {
  writeLock();
  try {
    map_.erase(id);

    file_.close();
    file_.open(FILE_PATH, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open master index during deletion");
    }

    for (const auto &[rec_id, rec_pos] : map_) {
      file_ << rec_id << ' ' << rec_pos << '\n';
    }
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

std::ios::pos_type MasterIndex::getPosition(int id) const {
  if (auto pos = map_.find(id); pos == map_.end()) {
    throw std::runtime_error("No such ID in master index");
  } else {
    return pos->second;
  }
}

void MasterIndex::updateFile() {
  readLock();
  try {
    map_.clear();

    file_.seekg(0, std::ios::beg);
    while (true) {
      int id;
      file_ >> id;
      if (file_.eof()) {
        file_.clear();
        break;
      }
      if (file_.fail() || id < 0) {
        throw std::domain_error("Invalid id information");
      }

      std::streamoff record_pos;
      file_ >> record_pos;
      if (file_.fail()) {
        throw std::domain_error("Invalid record position information");
      }

      map_[id] = record_pos;
    }
  } catch (std::exception &exception) {
    unlock();
    throw std::domain_error(exception.what());
  }
  unlock();
}

} // prod
