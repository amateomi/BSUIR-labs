#include "rwlock_manager.hpp"

#include <stdexcept>

#include <cstring>
#include <csignal>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

RwlockManager::RwlockManager(std::string_view shared_memory_lock)
    : shared_memory_(shared_memory_lock) {
  shm_unlink(shared_memory_.c_str());
  auto fd = shm_open(shared_memory_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    throw std::logic_error("shm_open: " + std::string{strerror(errno)});
  }

  if (ftruncate(fd, sizeof(pthread_rwlock_t))) {
    throw std::logic_error("ftruncate: " + std::string{strerror(errno)});
  }

  if (auto ptr = mmap(nullptr, sizeof(pthread_rwlock_t),
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      ptr == MAP_FAILED) {
    throw std::logic_error("mmap: " + std::string{strerror(errno)});
  } else {
    lock_ = reinterpret_cast<pthread_rwlock_t*>(ptr);
  }

  if (close(fd) < 0) {
    throw std::logic_error("close: " + std::string{strerror(errno)});
  }

  pthread_rwlockattr_t attr;
  if (pthread_rwlockattr_init(&attr)) {
    throw std::logic_error("Failed to initialize rwlock attribute");
  }
  if (pthread_rwlockattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) {
    throw std::logic_error("Failed to set shared rwlock attribute");
  }
  if (pthread_rwlock_init(lock_, &attr)) {
    throw std::logic_error("Failed to initialize rwlock");
  }

  if (sigemptyset(&sigio_) < 0) {
    throw std::runtime_error("sigemptyset: " + std::string{strerror(errno)});
  }
  if (sigaddset(&sigio_, SIGIO) < 0) {
    throw std::runtime_error("sigaddset: " + std::string{strerror(errno)});
  }
}

RwlockManager::~RwlockManager() {
  pthread_rwlock_destroy(lock_);
}

void RwlockManager::readLock() {
  if (was_rdlock_) {
    return;
  }
  if (sigprocmask(SIG_BLOCK, &sigio_, nullptr) < 0) {
    throw std::runtime_error("sigprocmask: " + std::string{strerror(errno)});
  }
  if (pthread_rwlock_rdlock(lock_)) {
    throw std::runtime_error("Failed to read lock");
  }
  was_rdlock_ = true;
}

void RwlockManager::writeLock() {
  if (sigprocmask(SIG_BLOCK, &sigio_, nullptr) < 0) {
    throw std::runtime_error("sigprocmask: " + std::string{strerror(errno)});
  }
  if (pthread_rwlock_wrlock(lock_)) {
    throw std::runtime_error("Failed to write lock");
  }
}

void RwlockManager::unlock() {
  if (was_rdlock_) {
    was_rdlock_ = false;
  }
  if (pthread_rwlock_unlock(lock_)) {
    throw std::runtime_error("Failed to unlock");
  }
  if (sigprocmask(SIG_UNBLOCK, &sigio_, nullptr) < 0) {
    throw std::runtime_error("sigprocmask: " + std::string{strerror(errno)});
  }
}
