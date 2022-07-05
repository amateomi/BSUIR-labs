#include <cstring>

#include "Stack/Stack.hpp"

int main() {
    Stack<double, 5> stack1;
    
    stack1.push(3.3);
    stack1.push(4.4);
    stack1.push(5.5);
    stack1.push(6.6);
    stack1.push(7.7);
    
    std::cout << "stack1:" << std::endl;
    print(stack1);
    
    stack1.pop();
    stack1.pop();
    stack1.pop();
    
    std::cout << "After 3 pop():" << std::endl;
    print(stack1);
    
    Stack<double, 3> stack2;
    
    stack2.push(3.3);
    stack2.push(4.4);
    
    std::cout << "stack2:" << std::endl;
    print(stack2);
    
    std::cout << "stack1 == stack2: " << std::boolalpha << (stack1 == stack2) << std::endl
              << "stack1 != stack2: " << std::boolalpha << (stack1 != stack2) << std::endl;
    
    stack2.push(5.5);
    std::cout << "5.5 pushed in stack2" << std::endl;
    
    std::cout << "stack1 == stack2: " << std::boolalpha << (stack1 == stack2) << std::endl
              << "stack1 != stack2: " << std::boolalpha << (stack1 != stack2) << std::endl;
    
    stack1.erase();
    std::cout << "stack1 is erased" << std::endl;
    
    stack1.push(2.2);
    stack1.push(1.1);
    
    std::cout << "new stack1:" << std::endl;
    print(stack1);
    
    std::cout << "stack2:" << std::endl;
    print(stack2);
    
    std::cout << "stack1 + stack2:" << std::endl;
    print((stack1 + stack2));
    
    std::cout << "stack2 + stack1:" << std::endl;
    print((stack2 + stack1));
    std::cout << "stack2 + stack1 compile time maxSize: " << (stack1 + stack2).getMaxStackSize() << std::endl;
    
    //##################################################################################################################
    Stack<char *, 4> stack3;
    
    char str1[] = "Call of Cthulhu";
    stack3.push(str1);
    
    char str2[] = "Call of Duty";
    stack3.push(str2);
    
    char str3[] = "Call of Chernobyl";
    stack3.push(str3);
    
    char str4[] = "What \"Call of\" you know?";
    stack3.push(str4);
    
    std::cout << "print strings specialisation:" << std::endl;
    print(stack3);

    //----------------------------Additional task----------------------------//

    stack1.push(6.7);
    stack1.push(-6.7);
    stack1.push(20.1);

    std::cout << "Stack1:" << std::endl;
    print(stack1);

    stack1.sort();
    std::cout << "Sorted stack1:" << std::endl;
    print(stack1);

    stack1.find(6.7);
    stack1.find(0);
}