#pragma once

#include <string>

#include <pthread.h>

class RwlockManager {
 public:
  explicit RwlockManager(std::string_view shared_memory_lock);
  ~RwlockManager();

  void readLock();
  void writeLock();
  void unlock();

 private:
  const std::string shared_memory_;

  pthread_rwlock_t* lock_{};
  sigset_t sigio_{};
  bool was_rdlock_{};
};
