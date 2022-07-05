#include "terminate-and-unexpected.hpp"
#include "Stack/Stack.hpp"
#include "input.hpp"

int main() {
    // Terminate example
    std::set_terminate(terminate);
//    throw std::runtime_error("qwe");

    // Unexpected example
    std::set_unexpected(unexpected);
//    try {
//        fun();
//    } catch (int errorCode) {
//        std::cout << "Error: " << errorCode << std::endl;
//    }


    Stack<int> stack;

    while (true) {
        std::cout << "Choose option:\n"
                     "1) - push\n"
                     "2) - pop\n"
                     "3) - print\n"
                     "4) - erase\n"
                     "0) - exit\n"
                     ">";

        switch (inputPositiveIntInRange(0, 4)) {
            case 1:
                std::cout << "Enter integer number:";
                try {
                    stack.push(inputInt());
                } catch (StackException &error) {
                    std::cerr << "Error: " << error.what() << std::endl;
                }
                break;

            case 2:
                try {
                    std::cout << stack.pop() << " popped" << std::endl;
                } catch (StackException &error) {
                    std::cerr << "Error: " << error.what() << std::endl;
                }
                break;

            case 3:
                stack.print();
                break;

            case 4:
                stack.erase();
                std::cout << "erased succeed" << std::endl;
                break;

            case 0:
                exit(EXIT_SUCCESS);
        }
    }
}