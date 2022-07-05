#include "directory_notify_manager.hpp"

#include <cstring>
#include <csignal>

#include <fcntl.h>

DirectoryNotifyManager::DirectoryNotifyManager(const fs::path &dir, void (* handler)(int)) {
  fd = open(dir.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::logic_error("open: " + std::string{strerror(errno)} + " (" + dir.c_str() + ")");
  }
  resetNotify(handler);
}

DirectoryNotifyManager::~DirectoryNotifyManager() {
  close(fd);
}

void DirectoryNotifyManager::resetNotify(void (* handler)(int)) const {
  if (fcntl(fd, F_NOTIFY, DN_MODIFY | DN_MULTISHOT) < 0) {
    throw std::logic_error("fcntl: " + std::string{strerror(errno)});
  }
  if (signal(SIGIO, handler) == SIG_ERR) {
    throw std::logic_error("signal: " + std::string{strerror(errno)});
  }
}
