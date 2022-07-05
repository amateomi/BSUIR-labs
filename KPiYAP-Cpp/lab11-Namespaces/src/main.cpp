#include <vector>

#include "input.hpp"
#include "print_edition.hpp"

int main() {
  using namespace std;
  using namespace book;

  vector<PrintEdition> vector;

  while (true) {
    try {
      std::cout << "Choose option:\n"
                   "1) - add print edition\n"
                   "2) - print\n"
                   "3) - move vector part\n"
                   "4) - remove copies\n"
                   "5) - find subset\n"
                   "0) - exit\n"
                   ">";

      switch (InputPositiveIntInRange(0, 7)) {
        case 1: {
          cout << "How many pages?\n";
          int pages_amount = InputPositiveInt();
          cin.ignore(100, '\n');
          std::vector<Page> pages;
          for (int i = 0; i < pages_amount; ++i) {
            cout << "Enter page text:\n";
            string text;
            getline(cin, text, '\n');
            pages.emplace_back(i + 1, text);
          }

          cout << "Enter book title:\n";
          string author_title;
          getline(cin, author_title, '\n');
          Cover cover(author_title);

          cout << "Enter binding type (hard - 1, soft - 2)\n";
          int type = InputPositiveIntInRange(1, 2);
          cin.ignore(100, '\n');
          Binding
              binding(type == 1 ? Binding::Type::HARD : Binding::Type::SOFT);

          cout << "Enter author full name:\n";
          string full_name;
          getline(cin, full_name, '\n');
          auto author = make_shared<Author>(full_name);

          cout << "Enter price:\n";
          int price = InputPositiveInt();
          cin.ignore(100, '\n');

          cout << "Enter publisher title:\n";
          string publisher_title;
          getline(cin, publisher_title, '\n');
          auto publisher = make_shared<Publisher>(publisher_title);

          vector.emplace_back(pages, cover, binding, author, price, publisher);

          cout << "Book added";
          break;
        }

        case 2: {
          for (int i = 0; i < vector.size(); ++i) {
            cout << "\nBook " << i << endl << vector[i];
          }
          break;
        }

        case 3: {
          // Get left value
          cout << "Enter left index:";
          int left_index = InputPositiveIntInRange(0, (int) vector.size() - 1);

          // Get right value
          cout << "Enter right index:";
          int right_index = InputPositiveIntInRange(0, (int) vector.size() - 1);

          // Range check
          if (left_index >= right_index) {
            throw runtime_error("Left index must be less than right index");
          }

          // Remember selected part
          std::vector<PrintEdition> temp;
          for (int i = left_index; i <= right_index; ++i) {
            temp.push_back(vector[i]);
          }

          // Erase vector
          vector.erase(vector.begin() + left_index,
                       vector.begin() + right_index + 1);

          // Print remaining
          cout << "Current vector:\n";
          for (int i = 0; i < vector.size(); ++i) {
            cout << "\nBook " << i << endl << vector[i];
          }

          // Insert
          cout << "Enter index to insert:\n";
          int insert = InputPositiveIntInRange(0, (int) vector.size() - 1);
          vector.insert(vector.begin() + insert, temp.begin(), temp.end());

          cout << "Part moved\n";
          break;
        }

        case 4: {
          for (int i = 0; i < vector.size(); ++i) {
            for (int j = i + 1; j < vector.size(); ++j) {
              // Check pages
              bool is_same_pages = true;
              if (vector[i].GetPages().size() == vector[j].GetPages().size()) {
                for (int k = 0; k < vector[i].GetPages().size(); ++k) {
                  if (vector[i].GetPages()[k] != vector[j].GetPages()[k]) {
                    is_same_pages = false;
                    break;
                  }
                }
              } else {
                is_same_pages = false;
              }

              if (is_same_pages &&
                  vector[i].GetPrice() == vector[j].GetPrice() &&
                  vector[i].GetBinding() == vector[j].GetBinding() &&
                  vector[i].GetCover() == vector[j].GetCover() &&
                  *vector[i].GetAuthor() == *vector[j].GetAuthor() &&
                  *vector[i].GetPublisher() == *vector[j].GetPublisher()) {
                vector.erase(vector.begin() + j);
                --j;
              }
            }
          }
          break;
        }

        case 5: {
          cout << "Select field:\n"
                  "1) - cover\n"
                  "2) - binding\n"
                  "3) - author\n"
                  "4) - price\n"
                  "5) - publisher\n"
                  ">";
          int field = InputPositiveIntInRange(1, 6);

          cout << "Enter elements amount:\n";
          int amount = InputPositiveInt();
          cin.ignore(100, '\n');

          switch (field) {
            case 1: {
              std::vector<Cover> keys;

              for (int i = 0; i < amount; ++i) {
                cout << "Enter book title:\n";
                string title;
                getline(cin, title, '\n');
                keys.emplace_back(title);
              }

              std::vector<PrintEdition> subset;
              for (int i = 0; i < vector.size(); ++i) {
                for (auto &key: keys) {
                  if (vector[i].GetCover() == key) {
                    subset.push_back(vector[i]);
                    ++i;
                  } else {
                    subset.clear();
                    break;
                  }
                }
                if (!subset.empty()) {
                  for (auto &item: subset) {
                    cout << item << endl;
                  }
                }
              }
              break;
            }

            case 2: {
              std::vector<Binding> keys;

              for (int i = 0; i < amount; ++i) {
                cout << "Enter binding type (hard - 1, soft - 2)\n";
                int type = InputPositiveIntInRange(1, 2);
                cin.ignore(100, '\n');
                keys.emplace_back(
                    type == 1 ? Binding::Type::HARD : Binding::Type::SOFT);
              }

              std::vector<PrintEdition> subset;
              for (int i = 0; i < vector.size(); ++i) {
                for (auto &key: keys) {
                  if (vector[i].GetBinding() == key) {
                    subset.push_back(vector[i]);
                    ++i;
                  } else {
                    subset.clear();
                    break;
                  }
                }
                if (!subset.empty()) {
                  for (auto &item: subset) {
                    cout << item << endl;
                  }
                }
              }
              break;
            }

            case 3: {
              std::vector<Author> keys;

              for (int i = 0; i < amount; ++i) {
                cout << "Enter author full name:\n";
                string full_name;
                getline(cin, full_name, '\n');
                keys.emplace_back(full_name);
              }

              std::vector<PrintEdition> subset;
              for (int i = 0; i < vector.size(); ++i) {
                for (auto &key: keys) {
                  if (*vector[i].GetAuthor() == key) {
                    subset.push_back(vector[i]);
                    ++i;
                  } else {
                    subset.clear();
                    break;
                  }
                }
                if (!subset.empty()) {
                  for (auto &item: subset) {
                    cout << item << endl;
                  }
                }
              }
              break;
            }

            case 4: {
              std::vector<int> keys;

              for (int i = 0; i < amount; ++i) {
                cout << "Enter price:\n";
                keys.emplace_back(InputPositiveInt());
                cin.ignore(100, '\n');
              }

              std::vector<PrintEdition> subset;
              for (int i = 0; i < vector.size(); ++i) {
                for (auto &key: keys) {
                  if (vector[i].GetPrice() == key) {
                    subset.push_back(vector[i]);
                    ++i;
                  } else {
                    subset.clear();
                    break;
                  }
                }
                if (!subset.empty()) {
                  for (auto &item: subset) {
                    cout << item << endl;
                  }
                }
              }
              break;
            }

            case 5: {
              std::vector<Publisher> keys;

              for (int i = 0; i < amount; ++i) {
                cout << "Enter publisher title:\n";
                string publisher_title;
                getline(cin, publisher_title, '\n');
                keys.emplace_back(publisher_title);
              }

              std::vector<PrintEdition> subset;
              for (int i = 0; i < vector.size(); ++i) {
                for (auto &key: keys) {
                  if (*vector[i].GetPublisher() == key) {
                    subset.push_back(vector[i]);
                    ++i;
                  } else {
                    subset.clear();
                    break;
                  }
                }
                if (!subset.empty()) {
                  for (auto &item: subset) {
                    cout << item << endl;
                  }
                }
              }
              break;
            }

            default:break;
          }
          break;
        }

        case 0:exit(EXIT_SUCCESS);
      }

    } catch (exception &error) {
      cout << "\nError: " << error.what() << endl;
    }
    system("pause>0");
    system("cls");
  }
}
