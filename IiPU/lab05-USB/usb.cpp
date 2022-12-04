#include <libudev.h>
#include <sys/mount.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <thread>

using namespace std;
using namespace filesystem;

enum class DeviceType {
    Drive = 1,
    Mouse = 2,
};

const auto MOUNT_LIST = "/proc/mounts";
const auto INTERFACE_TYPE = "bNumInterfaces";

class UsbDeviceManager {
public:
    UsbDeviceManager()
        : m_udev { udev_new() }
        , m_monitor { udev_monitor_new_from_netlink(m_udev, "udev") }
    {
        if (!m_udev) {
            cerr << "udev_new: failed" << endl;
            exit(EXIT_FAILURE);
        }
        udev_monitor_filter_add_match_subsystem_devtype(m_monitor, "usb", "usb_device");
        udev_monitor_enable_receiving(m_monitor);
    }

    ~UsbDeviceManager() { udev_unref(m_udev); }

    [[noreturn]] void observe()
    {
        while (true) {
            if (auto dev = udev_monitor_receive_device(m_monitor); dev) {

                string action { udev_device_get_action(dev) };
                string port { udev_device_get_sysname(dev) };
                string devicePath { udev_device_get_syspath(dev) };
                DeviceType deviceType {};
                path driveMountPath;

                if (action == "add") {
                    int type;
                    ifstream { devicePath + "/" + INTERFACE_TYPE } >> type;
                    deviceType = static_cast<DeviceType>(type);

                    if (deviceType == DeviceType::Drive) {
                        // Wait to sync with kernel
                        sleep(3);
                        ifstream mountList { MOUNT_LIST };

                        string lastLine, temp;
                        do {
                            lastLine = temp;
                            getline(mountList, temp);
                        } while (mountList);

                        istringstream iss { lastLine };
                        // Skip first word
                        iss >> m_drivePortToMount[port] >> m_drivePortToMount[port];
                        driveMountPath = m_drivePortToMount[port];
                    }
                } else if (action == "remove") {
                    deviceType = m_drivePortToMount.count(port) == 1 ? DeviceType::Drive : DeviceType::Mouse;
                    driveMountPath = m_drivePortToMount[port];
                    m_drivePortToMount.erase(port);
                }

                if (action == "add" || action == "remove") {
                    cout << endl
                         << "Action: " << action << endl
                         << "Port name: " << port << endl
                         << "Device path: " << devicePath << endl;
                    if (deviceType == DeviceType::Drive)
                        cout << "Drive mount path: " << driveMountPath << endl;
                    cout << endl;
                }
                udev_device_unref(dev);
            }
        }
    }

    void unmount(const string& port, int flag)
    {
        auto res = umount2(m_drivePortToMount[port].c_str(), flag);
        if (res < 0 && errno == EBUSY) {
            cout << "USB drive is busy right now" << endl;
        } else if (res < 0) {
            perror("umount2");
            exit(errno);
        }
    }

private:
    udev* m_udev {};
    udev_monitor* m_monitor {};
    map<string, path> m_drivePortToMount;
};

int main()
{
    if (getegid()) {
        cerr << "Rerun program as root!" << endl;
        exit(EXIT_FAILURE);
    }
    UsbDeviceManager manager;
    thread observer { &UsbDeviceManager::observe, &manager };
    while (true) {
        cout << "Enter option: " << endl
             << "1) - Safe unmount" << endl
             << "2) - Unsafe unmount" << endl
             << "q) - Quit" << endl;
        string option;
        getline(cin, option);
        if (option == "1" || option == "2") {
            cout << "Enter port name to unmount drive: " << endl;
            string port;
            getline(cin, port);
            if (option == "1") {
                manager.unmount(port, 0);
            } else {
                manager.unmount(port, MNT_DETACH);
            }
        } else if (option == "q") {
            exit(EXIT_SUCCESS);
        }
    }
}
