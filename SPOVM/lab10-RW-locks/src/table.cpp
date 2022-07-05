#include "table.hpp"

#include <cstring>
#include <csignal>

static prod::Table* table;

void notify_handler(int) {
  table->updateFile();
  table->resetNotify(notify_handler);
}

namespace prod {

Table::Table(const fs::path &file_path)
    : RwlockManager("/table"),
      FileManager(file_path),
      DirectoryNotifyManager(file_path.parent_path(), notify_handler),
      deletion_accounting_(file_path.parent_path().append("deletion_accounting")),
      master_index_(file_path.parent_path().append("master_index")),
      name_index_(file_path.parent_path().append("name_index")),
      code_index_(file_path.parent_path().append("code_index")){

  table = this;
  if (signal(SIGIO, notify_handler) == SIG_ERR) {
    throw std::logic_error("signal: " + std::string{strerror(errno)});
  }

  readLock();
  try {
    std::string buffer;
    std::getline(file_, buffer, '\n');
    if (buffer != "HEADER") {
      throw std::domain_error("No header in file");
    }

    file_ >> buffer;
    seed_file_pos_ = file_.tellg();
    int seed;
    file_ >> seed;
    if (file_.fail() || buffer != "seed" || seed < 0) {
      throw std::domain_error("Invalid seed information");
    }
    seed_file_pos_ += 1;

    file_ >> buffer;
    amount_file_pos_ = file_.tellg();
    int amount;
    file_ >> amount;
    if (file_.fail() || buffer != "amount" || amount < 0) {
      throw std::domain_error("Invalid amount information");
    }
    amount_file_pos_ += 1;

    file_ >> buffer;
    if (file_.fail() || buffer != "TABLE") {
      throw std::domain_error("No table in file");
    }
    file_.ignore(1);

    std::getline(file_, buffer, '\n');
    if (file_.fail() || buffer != Record::TABLE_FORMAT) {
      throw std::domain_error("No columns in file");
    }

  } catch (std::exception &exception) {
    unlock();
    throw std::domain_error(exception.what());
  }
  unlock();
}

void Table::addRecord(Record &record) {
  writeLock();
  try {
    if (!code_index_.isAvailableCode(record.code)) {
      throw std::runtime_error("Code is not unique");
    }

    file_.seekg(seed_file_pos_);
    int seed;
    file_ >> seed;
    ++seed;
    record.id = seed;
    file_.seekp(seed_file_pos_);
    file_ << std::setfill('0') << std::setw(10) << std::right << seed;

    file_.seekg(amount_file_pos_);
    int amount;
    file_ >> amount;
    ++amount;
    file_.seekp(amount_file_pos_);
    file_ << std::setfill('0') << std::setw(10) << std::right << amount;

    file_.fill(' ');

    auto record_pos = deletion_accounting_.getEmptyPosition();
    if (record_pos < 0) {
      file_.seekg(0, std::ios::end);
      record_pos = file_.tellg();
    }
    file_.seekp(record_pos);
    file_ << record;
    file_.flush();

    master_index_.addRecord(record.id, record_pos);
    name_index_.addRecord(record.name, record.id);
    code_index_.addRecord(record.code, record.id);

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

void Table::delRecord(int id) {
  writeLock();
  try {
    auto pos = master_index_.getPosition(id);

    file_.seekg(pos);

    master_index_.delRecord(id);
    std::string name;
    file_ >> id >> name;
    name_index_.delRecord(name);
    int code;
    file_ >> code;
    code_index_.delRecord(code);

    deletion_accounting_.addDeletion(pos);

    file_.seekp(pos);
    file_ << "                                                            ";

    file_.seekg(amount_file_pos_);
    int amount;
    file_ >> amount;
    --amount;
    file_.seekp(amount_file_pos_);
    file_ << std::setfill('0') << std::setw(10) << std::right << amount;

    file_.fill(' ');
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

Record Table::getRecord(int id) {
  Record record;
  readLock();
  try {
    file_.seekg(master_index_.getPosition(id));
    file_ >> record.id >> record;

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
  return record;
}

void Table::putRecord(const Record &new_record) {
  writeLock();
  try {
    auto pos = master_index_.getPosition(new_record.id);

    file_.seekg(pos);
    Record old_record;
    file_ >> old_record.id >> old_record;

    if (old_record.code != new_record.code &&
        !code_index_.isAvailableCode(new_record.code)) {
      throw std::runtime_error("Code is not unique");
    }

    if (old_record.name != new_record.name) {
      name_index_.delRecord(old_record.name);
      name_index_.addRecord(new_record.name, new_record.id);
    }
    if (old_record.code != new_record.code) {
      code_index_.delRecord(old_record.code);
      code_index_.addRecord(new_record.code, new_record.id);
    }

    file_.seekp(pos);
    file_ << new_record;
    file_.flush();

  } catch (std::exception &exception) {
    unlock();
    throw std::runtime_error(exception.what());
  }
  unlock();
}

std::variant<std::list<int>, int> Table::getPrimary(std::variant<std::string_view, int> index) const {

  if (std::holds_alternative<std::string_view>(index)) {
    return name_index_.getId(std::get<std::string_view>(index));

  } else {
    return code_index_.getId(std::get<int>(index));
  }
}

void Table::updateFile() {
  deletion_accounting_.updateFile();
  master_index_.updateFile();
  name_index_.updateFile();
  code_index_.updateFile();
}

} // prod
