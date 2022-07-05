#include <iostream>
#include <vector>
#include <algorithm>

#include "input.hpp"
#include "singly_linked_list.hpp"

int main() {
  using namespace std;

  vector<List<int, true>> array;

  while (true) {
    try {
      std::cout << "Choose option:\n"
                   "1) - add number\n"
                   "2) - add list\n"
                   "3) - print\n"
                   "4) - sort lists\n"
                   "5) - move list part\n"
                   "6) - find subset\n"
                   "0) - exit\n"
                   ">";

      switch (inputPositiveIntInRange(0, 7)) {

        case 1: {
          cout << "Enter list number:";
          int list_number = inputPositiveInt();
          cout << "Enter number:";
          array.at(list_number).Add(array.at(list_number).begin(), inputInt());
          cout << "Number added\n";
          break;
        }

        case 2: {
          array.emplace_back(List<int, true>());
          cout << "List added\n";
          break;
        }

        case 3: {
          int count = 0;
          for (auto &item: array) {
            cout << "List " << count << ":\n";
            item.Print();
            count++;
          }
          break;
        }

        case 4: {
          cout << "Ascending or descending (1/2):";
          int sort_type = inputPositiveIntInRange(1, 2);

          // Count total value
          vector<int> list_total_values;
          int index = 0;
          for (auto &item: array) {
            list_total_values.emplace_back(0);
            for (auto &value: item) {
              list_total_values.at(index) += value;
            }
            index++;
          }

          // Sort
          for (int i = 0; i < array.size(); ++i) {
            for (int j = i + 1; j < array.size(); ++j) {
              if ((sort_type == 1 &&
                  list_total_values.at(i) > list_total_values.at(j)) ||
                  (sort_type == 2 &&
                      list_total_values.at(i) < list_total_values.at(j))) {
                auto temp = array.at(i).head_;
                array.at(i).head_ = array.at(j).head_;
                array.at(j).head_ = temp;
              }
            }
          }
          cout << "Sorted\n";
          break;
        }

        case 5: {
          // Get list number
          cout << "Enter list number:";
          int list_number = inputPositiveInt();
          cout << "Current list:";
          array.at(list_number).Print();

          // Get left value
          cout << "Enter left value:";
          int left_value = inputInt();
          auto left_iter = array.at(list_number).Find(left_value);
          if (left_iter == array.at(list_number).end()) {
            throw runtime_error("No such left value");
          }

          // Get right value
          cout << "Enter right value:";
          int right_value = inputInt();
          auto right_iter = array.at(list_number).Find(right_value);
          if (right_iter == array.at(list_number).end()) {
            throw runtime_error("No such right value");
          }

          // Range check
          auto it = left_iter;
          while (it != right_iter) {
            if (it == array.at(list_number).end()) {
              throw runtime_error("Invalid range");
            }
            ++it;
          }

          // Remember selected part
          auto temp_list = new List<int, true>;
          temp_list->Add(temp_list->end(), *left_iter);
          auto temp_list_last_iter = temp_list->begin();

          it = left_iter;
          do {
            ++it;
            temp_list->Add(temp_list->end(), *it);
            ++temp_list_last_iter;
          } while (it != right_iter);

          // Erase selected part
          it = left_iter;
          do {
            auto it_prev = it;
            ++it;
            array.at(list_number).Delete(it_prev);
          } while (it != right_iter);
          if (right_iter != array.at(list_number).end()) {
            array.at(list_number).Delete(right_iter);
          }

          // Print remaining part
          cout << "Current list:";
          array.at(list_number).Print();
          cout << "Erased part:";
          temp_list->Print();

          // Insert in head
          cout << "Insert in head? 1 - yes, 2 - no\n";
          if (inputPositiveIntInRange(1, 2) == 1) {
            temp_list_last_iter.node_ptr_->next_ = array.at(list_number).head_;
            array.at(list_number).head_ = temp_list->head_;
          } else {
            // Get value before insert position
            cout << "Enter value before insert position:";
            int insert_value = inputInt();
            auto insert_iter = array.at(list_number).Find(insert_value);
            if (insert_iter == array.at(list_number).end()) {
              throw runtime_error("No such insert value");
            }

            // Insert sublist
            temp_list_last_iter.node_ptr_->next_ = insert_iter.node_ptr_->next_;
            insert_iter.node_ptr_->next_ = temp_list->head_;
          }
          cout << "Part moved\n";
          break;
        }

        case 6: {
          List<int, true> subset;

          // Get subset
          do {
            cout << "Enter number:";
            subset.Add(subset.end(), inputInt());
            cout << "Continue? 1 - yes, 2 - no\n";
          } while (inputInt() == 1);

          // Search subset
          int list_number = 0;
          for (auto &list: array) {
            bool isSubset = false;
            auto subset_iter = subset.begin();
            for (auto &value: list) {
              if (value == *subset_iter) {
                isSubset = true;
                ++subset_iter;
                if (subset_iter == subset.end()) {
                  cout << "List " << list_number << ' ';
                  array.at(list_number).Print();
                  cout << "has subset ";
                  subset.Print();
                  break;
                }
              } else if (isSubset) {
                break;
              }
            }
            list_number++;
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