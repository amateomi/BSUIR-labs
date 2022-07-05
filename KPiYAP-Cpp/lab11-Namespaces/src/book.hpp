  //
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#ifndef LAB11_SRC_BOOK_HPP_
#define LAB11_SRC_BOOK_HPP_

#include <iostream>
#include <vector>
#include <utility>
#include <memory>

#include "page.hpp"
#include "cover.hpp"
#include "binding.hpp"
#include "author.hpp"

namespace book {

class Book {
 public:
  // Constructors
  Book();
  Book(std::vector<Page> pages,
       const Cover &cover,
       const Binding &binding,
       std::shared_ptr<Author> author);

  // Destructor
  virtual ~Book();

  // Getters and setters
  [[nodiscard]]  const std::vector<Page> &GetPages() const;
  void SetPages(const std::vector<Page> &pages);

  [[nodiscard]]  const Cover &GetCover() const;
  void SetCover(const Cover &cover);

  [[nodiscard]]  const Binding &GetBinding() const;
  void SetBinding(const Binding &binding);

  [[nodiscard]] const std::shared_ptr<Author> &GetAuthor() const;
  void SetAuthor(const std::shared_ptr<Author> &author);

 protected:
  // Composition
  std::vector<Page> pages_;
  Cover cover_;
  Binding binding_;

  // Aggregation
  std::shared_ptr<Author> author_;
};

}

#endif //LAB11_SRC_BOOK_HPP_
