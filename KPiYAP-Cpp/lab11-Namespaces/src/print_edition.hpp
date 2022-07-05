//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_PRINT_EDITION_HPP_
#define LAB11_SRC_PRINT_EDITION_HPP_

#include <utility>
#include <memory>
#include <ostream>

#include "book.hpp"
#include "publisher.hpp"

namespace book {

class PrintEdition : public Book {
 public:
  // Constructors
  PrintEdition();
  PrintEdition(const std::vector<Page> &pages,
               const Cover &cover,
               const Binding &binding,
               const std::shared_ptr<Author> &author,
               int price,
               std::shared_ptr<Publisher> publisher);

  // Destructor
  ~PrintEdition() override;

  // Getters and setters
  [[nodiscard]] int GetPrice() const;
  void SetPrice(int price);

  [[nodiscard]] const std::shared_ptr<Publisher> &GetPublisher() const;
  void SetPublisher(const std::shared_ptr<Publisher> &publisher);

  // Printing
  friend
  std::ostream &operator<<(std::ostream &os, const PrintEdition &book);

 private:
  int price_{};

  // Aggregation
  std::shared_ptr<Publisher> publisher_;
};

}

#endif //LAB11_SRC_PRINT_EDITION_HPP_
