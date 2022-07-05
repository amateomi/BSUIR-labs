#include "name_index.hpp"

namespace prod {

NameIndex::NameIndex(const fs::path &file_path)
    : RwlockManager("/name_index"),
      FileManager(file_path) {
  updateFile();
}

void NameIndex::addRecord(std::string_view name, int id) {
  writeLock();
  try {
    file_.seekp(0, std::ios::end);
    file_ << name << ' ' << id << '\n';
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

void NameIndex::delRecord(std::string_view name) {
  writeLock();
  try {
    map_.erase(name.data());

    file_.close();
    file_.open(FILE_PATH, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      throw std::runtime_error("Failed to open name index during deletion");
    }

    for (const auto &[rec_name, rec_id] : map_) {
      file_ << rec_name << ' ' << rec_id << '\n';
    }
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

std::list<int> NameIndex::getId(std::string_view name) const {
  if (map_.find(name.data()) == map_.end()) {
    throw std::runtime_error("No such name in name index");
  }
  std::list<int> ids;
  for (auto [iter, iter_end] = map_.equal_range(name.data());
       iter != iter_end;
       ++iter) {
    ids.emplace_back(iter->second);
  }
  return ids;
}

void NameIndex::updateFile() {
  readLock();
  try {
    map_.clear();

    file_.seekg(0, std::ios::beg);
    while (true) {
      std::string name;
      file_ >> name;
      if (file_.eof()) {
        file_.clear();
        break;
      }
      if (file_.fail() || name.size() > 16) {
        throw std::domain_error("Invalid name information");
      }

      int id;
      file_ >> id;
      if (file_.fail() || id < 0) {
        throw std::domain_error("Invalid id information");
      }

      map_.insert({name, id});
    }
  } catch (std::exception &exception) {
    unlock();
    throw std::domain_error(exception.what());
  }
  unlock();
}

} // prod
