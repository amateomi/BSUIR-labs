//
// Created by amaterasu on 17.11.2021.
//

#ifndef LAB10_SRC_SINGLY_LINKED_LIST_HPP_
#define LAB10_SRC_SINGLY_LINKED_LIST_HPP_

#include <iterator>
#include <ostream>

template<typename T, bool unique>
class List {
 public:
  // Simple struct-like node for my 'List'
  class Node {
   public:
    T data_;
    Node *next_{0};

    // Create node
    explicit Node(T val) : data_(val), next_(nullptr) {}
    // Delete 'next_', cause chain of erasing
    ~Node() { delete next_; }
  };

  // List forward iterator
  class Iterator {
   public:
    // Constructors
    Iterator() : node_ptr_(nullptr) {}
    explicit Iterator(Node *ptr) : node_ptr_(ptr) {}

    // Copy constructor
    Iterator(const Iterator &it) : node_ptr_(it.node_ptr_) {}
    // Overload '='
    Iterator &operator=(const Iterator &it) noexcept {
      if (this == &it) {
        return *this;
      }
      node_ptr_ = it.node_ptr_;
      return *this;
    }

    // Overload equality operators
    bool operator==(const Iterator &it) const noexcept {
      return node_ptr_ == it.node_ptr_;
    }
    bool operator!=(const Iterator &it) const noexcept {
      return node_ptr_ != it.node_ptr_;
    }

    // Set to the next Node
    Iterator &operator++() {
      if (!node_ptr_) {
        throw std::runtime_error("Incremented an empty iterator");
      }
      node_ptr_ = node_ptr_->next_;
      return *this;
    }

    // Get value
    T &operator*() {
      if (!node_ptr_) {
        throw std::runtime_error("Tried to dereference an empty iterator");
      }
      return node_ptr_->data_;
    }

   public:
    friend class List;
    Node *node_ptr_;
  };

  // Deleted methods
  List &operator=(const List &) = delete;

  // Constructors and destructors
  List() : head_(nullptr) {}
  List(const List &list) {
    for (auto &item: list) {
      Add(end(), item);
    }
  }

  virtual ~List() { delete head_; }

  // Iterator utilities
  Iterator begin() const noexcept { return Iterator(head_); }
  Iterator end() const noexcept { return Iterator(); }

  // Add item in List on specific position
  void Add(const Iterator &pos, const T &val) {
    // Unique check
    if (unique && IsInList(val)) return;
    // Add to front
    if (pos == begin()) {
      auto new_head = new Node(val);
      new_head->next_ = head_;
      head_ = new_head;
      return;
    }

    // Check all list
    for (auto it = begin(); it != end(); ++it) {
      // If next Node is new_node position
      if (it.node_ptr_->next_ == pos.node_ptr_) {
        auto new_node = new Node(val);
        new_node->next_ = it.node_ptr_->next_;
        it.node_ptr_->next_ = new_node;
        return;
      }
    }
    throw std::runtime_error("Invalid iterator position");
  }

  // Return iterator position of the element
  // Return end() if no such 'val'
  Iterator Find(const T &val) const noexcept {
    for (auto it = begin(); it != end(); ++it) {
      if (*it == val) return it;
    }
    // If no matching
    return end();
  }

  void Delete(const Iterator &pos) {
    if (pos == end()) throw std::runtime_error("Invalid iterator position");
    // Delete head case
    if (pos == begin()) {
      auto prev_head = pos.node_ptr_;
      head_ = head_->next_;
      // Set 'next_' to nullptr to prevent delete chain
      prev_head->next_ = nullptr;
      delete prev_head;
    } else {
      // Check all list
      for (auto it = begin(); it != end(); ++it) {
        if (it.node_ptr_->next_ == pos.node_ptr_) {
          auto node = pos.node_ptr_;
          it.node_ptr_->next_ = pos.node_ptr_->next_;
          // Set 'next_' to nullptr to prevent delete chain
          node->next_ = nullptr;
          delete node;
        }
      }
    }
  }

  bool IsInList(const T &val) const noexcept {
    return Find(val) != end();
  }

  void Print() const noexcept {
    for (const auto &data: (*this)) {
      std::cout << data << ' ';
    }
    std::cout << std::endl;
  }

 public:
  Node *head_{0};
};

#endif //LAB10_SRC_SINGLY_LINKED_LIST_HPP_
