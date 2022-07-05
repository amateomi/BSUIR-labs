//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "book.hpp"

book::Book::Book() {
  std::cout << "Book: default" << std::endl;
}

book::Book::Book(std::vector<Page> pages,
                 const book::Cover &cover,
                 const book::Binding &binding,
                 std::shared_ptr<Author> author)
    : pages_(std::move(pages)),
      cover_(cover),
      binding_(binding),
      author_(std::move(author)) {
}

book::Book::~Book() {
}

const std::vector<book::Page> &book::Book::GetPages() const {
  return pages_;
}

void book::Book::SetPages(const std::vector<Page> &pages) {
  pages_ = pages;
}

const book::Cover &book::Book::GetCover() const {
  return cover_;
}

void book::Book::SetCover(const book::Cover &cover) {
  cover_ = cover;
}

const book::Binding &book::Book::GetBinding() const {
  return binding_;
}

void book::Book::SetBinding(const book::Binding &binding) {
  binding_ = binding;
}

const std::shared_ptr<book::Author> &book::Book::GetAuthor() const {
  return author_;
}

void book::Book::SetAuthor(const std::shared_ptr<Author> &author) {
  author_ = author;
}

