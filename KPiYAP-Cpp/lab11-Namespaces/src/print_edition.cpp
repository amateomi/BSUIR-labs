//
// Created by Andrey Sikorin
// BSUIR 19.11.2021
//

#include "print_edition.hpp"

book::PrintEdition::PrintEdition() {
  std::cout << "PrintEdition: default" << std::endl;
}

book::PrintEdition::PrintEdition(const std::vector<Page> &pages,
                                 const book::Cover &cover,
                                 const book::Binding &binding,
                                 const std::shared_ptr<Author> &author,
                                 int price,
                                 std::shared_ptr<Publisher> publisher)
    : Book(pages, cover, binding, author),
      price_(price),
      publisher_(std::move(publisher)) {
}

book::PrintEdition::~PrintEdition() {
}

int book::PrintEdition::GetPrice() const {
  return price_;
}

void book::PrintEdition::SetPrice(int price) {
  price_ = price;
}

const std::shared_ptr<book::Publisher> &book::PrintEdition::GetPublisher() const {
  return publisher_;
}

void book::PrintEdition::SetPublisher(const std::shared_ptr<Publisher> &publisher) {
  publisher_ = publisher;
}

std::ostream &book::operator<<(std::ostream &os,
                               const book::PrintEdition &book) {
  os << "Title: " << book.GetCover().GetTitle() << std::endl
     << "Author: " << book.GetAuthor()->GetFullName() << std::endl
     << "Publisher: " << book.GetPublisher()->GetTitle() << std::endl
     << "Binding type: " << book.GetBinding() << std::endl
     << "Price: " << book.GetPrice() << '$' << std::endl;
  std::cout << "Text:" << std::endl;
  for (const auto &page: book.GetPages()) {
    std::cout << "page number " << page.GetNumber() << std::endl
              << page.GetText() << std::endl;
  }
  return os;
}

