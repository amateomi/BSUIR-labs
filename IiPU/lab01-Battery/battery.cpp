#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace std;

void clearConsole()
{
    cout << "\033[2J\033[1;1H";
}

string makeRed(const string& message)
{
    return "\033[1;31m" + message + "\033[0m";
}

const auto batteryFilepath { "/sys/class/power_supply/BAT0/"s };
const auto powerModePath { "/sys/firmware/acpi/platform_profile"s };
const auto powerState { "/sys/power/state"s };

atomic<bool> didEnterOption { false };

void batteryMonitor()
{
    while (!didEnterOption) {
        clearConsole();
        cout << "Charging status:\t" << ifstream { batteryFilepath + "status" }.rdbuf()
             << "Battery type:\t\t" << ifstream { batteryFilepath + "technology" }.rdbuf()
             << "Battery percentage:\t" << ifstream { batteryFilepath + "capacity" }.rdbuf()
             << "Power mode:\t\t" << ifstream { powerModePath }.rdbuf();

        cout << "\nExtra Task\n";

        string status;
        ifstream { batteryFilepath + "status" } >> status;
        if (status == "Discharging") {
            ifstream energyFile { batteryFilepath + "energy_now" };
            int energy = 0;
            energyFile >> energy;
            ifstream powerFile { batteryFilepath + "power_now" };
            int power = 0;
            powerFile >> power;

            const double floatTime = static_cast<double>(energy) / power;
            const int hours = static_cast<int>(floatTime);
            const int minutes = static_cast<int>((floatTime - hours) * 60);

            cout << "Battery remaining time:\t" << hours << ':' << setw(2) << setfill('0') << minutes << '\n';
        } else
            cout << makeRed("Battery charging\n");

        cout << "\nEnter option:\n"
             << "s) - Suspend\n"
             << "h) - Hibernate\n"
             << '>';
        cout.flush();

        this_thread::sleep_for(1s);
    }
}

int main()
{
    if (getegid()) {
        cerr << makeRed("Rerun program as root!") << endl;
        exit(EXIT_FAILURE);
    }
    thread monitor { batteryMonitor };

    string option;
    while (true) {
        cin >> option;
        if (cin.good() && (option == "s" || option == "h"))
            break;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << makeRed("Invalid input! Try again.") << endl;
    }

    didEnterOption = true;
    monitor.join();

    // Require root privileges
    if (option == "s")
        ofstream { powerState, ios::app } << "mem";
    else if (option == "h")
        ofstream { powerState, ios::app } << "disk";
}
