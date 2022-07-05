#include <cassert>
#include <cctype>

#include <iostream>

using namespace std;

// Get C-style string from std::cin and check for validation
bool getName(char *name, int nameLengthMax);

// Get C-style string from std::cin and check for validation
bool getLastname(char *lastname, int lastnameLengthMax);

// Get 'y' or 'n' form std::cin and convert in into boolean value
bool getYesOrNo();

// Get positive integer from std::cin and check for validation
int getNumber();

int main() {
    const int STRING_LENGTH = 80;

    // Allocating memory for name
    char *name;
    name = (char *) malloc((STRING_LENGTH + 1) * sizeof(char));
    assert(name && "Error: memory allocation.");

    // Allocating memory for name
    char *lastname;
    lastname = (char *) malloc((STRING_LENGTH + 1) * sizeof(char));
    assert(lastname && "Error: memory allocation.");

    // Get name form user, it must be single word form letters
    do {
        cout << "Enter your name:" << endl;
    } while (!getName(name, STRING_LENGTH));

    // Get lastname form user, it might be two words form letters divided with '-'
    do {
        cout << "Enter your lastname:" << endl;
    } while (!getLastname(lastname, STRING_LENGTH));

    // Ask user some questions, punish him, if answer is wrong
    // Question №1
    while (true) {
        cout << name << ' ' << lastname << ", do you like C++ programming language? [y/n]" << endl;
        bool answer = getYesOrNo();
        if (answer) {
            cout << "Wrong answer! Answer again." << endl;
        } else {
            cout << "Me too!" << endl;
            break;
        }
    }
    // Question №2
    while (true) {
        cout << name << ' ' << lastname << ", how many variation of initialisation in C++?" << endl;
        int answer = getNumber();
        if (answer != 3) {
            cout << "Wrong answer! Answer again." << endl;
        } else {
            cout << "Yes, you can use =, (), in C++11 {}" << endl;
            break;
        }
    }
    // Question №3
    cout << name << ' ' << lastname << ", is your name really " << name << "? [y/n]" << endl;
    bool answer = getYesOrNo();
    if (answer) {
        cout << "Ok." << endl;
    } else {
        cout << "Whatever man." << endl;
    }

    free(name);

    return 0;
}

bool getName(char *name, const int nameLengthMax) {
    char c;

    int i;
    for (i = 0; i < nameLengthMax; ++i) {
        cin.unsetf(ios::skipws);
        cin >> c;

        if (c == '\n') {
            break;
        } else if (islower(c) || isupper(c)) {
            // Check for letter.
            name[i] = c;
        } else {
            cerr << "Invalid data! Use only letters, enter only one word." << endl;
            cin.ignore(100, '\n');
            return false;
        }
    }
    name[i] = '\0';

    if (i == nameLengthMax && cin.peek() != '\n') {
        cout << "Your name too long, it is cut to " << name << endl;
    }

    return true;
}

bool getLastname(char *lastname, const int lastnameLengthMax) {
    bool wasDash = false;
    char c;

    int i;
    for (i = 0; i < lastnameLengthMax; ++i) {
        cin.unsetf(ios::skipws);
        cin >> c;

        if (c == '\n') {
            break;
        } else if (islower(c) || isupper(c)) {
            // Check for letter.
            lastname[i] = c;
        } else if (c == '-' && !wasDash) {
            wasDash = true;
            lastname[i] = c;
        } else {
            cerr << "Invalid data! Use only letters and single dash." << endl;
            cin.ignore(100, '\n');
            return false;
        }
    }
    lastname[i] = '\0';

    if (i == lastnameLengthMax && cin.peek() != '\n') {
        cout << "Your name too long, it is cut to " << lastname << endl;
    }

    return true;
}

bool getYesOrNo() {
    char c;
    while (true) {
        cin.unsetf(ios::skipws);
        cin >> c;

        bool isYesOrNo = (c == 'y' || c == 'n');
        if (!isYesOrNo || cin.peek() != '\n') {
            cin.ignore(100, '\n');
            cerr << "Invalid input! Enter only 'y' or 'n'." << endl;
        } else {
            cin.ignore(100, '\n');
            return c == 'y';
        }
    }
}

int getNumber() {
    int number;

    while (true) {
        cin >> number;

        if (cin.fail() || number <= 0 || cin.peek() != '\n') {
            cin.clear();
            cin.ignore(100, '\n');
            cerr << "Invalid input! Enter positive integer." << endl;
        } else {
            cin.ignore(100, '\n');
            return number;
        }
    }
}