#include <stdio.h>
#include <unistd.h>
#include <libudev.h>
#include <sys/mount.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/inotify.h>
#include <stdlib.h>
#include <cstring>

#include <algorithm>
#include <list>
#include <string>
#include <filesystem>
#include <iostream>
#include <future>
#include <map>
#include <fstream>
#include <thread>
#include <memory>
#include <future>

using namespace std;
using namespace filesystem;

constexpr auto MAX_EVENTS = 64;
constexpr auto DRIVE = 1;
constexpr auto MOUSE = 2;

const auto MOUNT_LIST = "/proc/mounts";
const auto INTERFACE_TYPE = "bNumInterfaces";
const auto MOUNT_DIR = "/run/media/" + string{getlogin()};

class SmartWatcher {
public:
    SmartWatcher(int inotifyFd, const path& path)
            : m_inotifyFd{inotifyFd},
              m_watcherFd{inotify_add_watch(inotifyFd, path.c_str(), IN_OPEN | IN_CLOSE)} {
        if (m_watcherFd < 0) {
            perror("add_watch");
            exit(EXIT_FAILURE);
        }
    }

    ~SmartWatcher() {
        inotify_rm_watch(m_inotifyFd, m_watcherFd);
    }

    void onEvent(const inotify_event& event) {
        if (event.wd == m_watcherFd) {
            if (event.mask & IN_OPEN) {
                ++m_openCloseCounter;
                cout << "Open event: " << m_watcherFd << ", counter: " << m_openCloseCounter << endl;
            }
            if (event.mask & IN_CLOSE) {
                --m_openCloseCounter;
                cout << "Close event: " << m_watcherFd << ", counter: " << m_openCloseCounter << endl;
            }
        }
    }

    [[nodiscard]] bool isSafeToUnmount() const { return m_openCloseCounter == 0; }

private:
    int m_inotifyFd{};
    int m_watcherFd{};
    int m_openCloseCounter{};
};

class UsbDeviceManager {
public:
    UsbDeviceManager()
            : m_udev{udev_new()},
              m_monitor{udev_monitor_new_from_netlink(m_udev, "udev")},
              m_inotifyFd{inotify_init()} {
        if (!m_udev) {
            cerr << "udev_new: failed" << endl;
            exit(EXIT_FAILURE);
        }
        udev_monitor_filter_add_match_subsystem_devtype(m_monitor, "usb", "usb_device");
        udev_monitor_enable_receiving(m_monitor);
    }

    ~UsbDeviceManager() {
        udev_unref(m_udev);
    }

    [[noreturn]] void observe() {
        while (true) {
            if (auto dev = udev_monitor_receive_device(m_monitor);
                    dev) {
                if (string action{udev_device_get_action(dev)};
                        action == "add" || action == "remove") {
                    string port{udev_device_get_sysname(dev)};
                    string devicePath{udev_device_get_syspath(dev)};

                    cout << endl
                         << "Action: " << action << endl
                         << "Port name: " << port << endl
                         << "Device path: " << devicePath << endl;

                    int deviceType{};
                    if (action == "add")
                        ifstream{devicePath + "/" + INTERFACE_TYPE} >> deviceType;
                    else if (action == "remove")
                        deviceType = m_drives.count(port) == 1 ? DRIVE : MOUSE;

                    if (deviceType == DRIVE) {
                        handleDriveAction(port, action);
                    }

                    cout << endl;
                }
                udev_device_unref(dev);
            }
        }
    }

    bool isSafeToUnmount(const string& port) {
        return all_of(m_drives[port].watchers.begin(),
                      m_drives[port].watchers.end(),
                      [](const auto& watcher) {
                          return watcher.isSafeToUnmount();
                      });
    }

    void unmount(const string& port) {
        auto res = umount(m_drives[port].mountPath.c_str());
        if (res < 0) {
            perror("umount");
            exit(EXIT_FAILURE);
        }
    }

private:
    /// Return mount path if USB drive
    void handleDriveAction(const string& port, const string& action) {
        if (action == "add") {
            // Wait to sync with kernel
            sleep(3);
            ifstream mountList{MOUNT_LIST};

            string lastLine, temp;
            do {
                lastLine = temp;
                getline(mountList, temp);
            } while (mountList);

            istringstream iss{lastLine};
            path mountPath;
            // Skip first word
            iss >> mountPath >> mountPath;
            m_drives[port].mountPath = mountPath;

            // Add notifiers
            for (const auto& file: directory_iterator{m_drives[port].mountPath})
                m_drives[port].watchers.emplace_back(m_inotifyFd, file.path());

            m_drives[port].eventHandler = new thread{&UsbDeviceManager::OnEvent, this, port};

            cout << "USB drive mount path: " << m_drives[port].mountPath << endl;

        } else if (action == "remove") {
            cout << "USB drive mount path: " << m_drives[port].mountPath << endl;
            m_drives[port].stopHandler = true;
            m_drives[port].eventHandler->join();
            delete m_drives[port].eventHandler;
            m_drives.erase(port);
        }
    }

    void OnEvent(const string& port) {
        inotify_event events[MAX_EVENTS];
        while (!m_drives[port].stopHandler) {
            memset(events, 0, sizeof(events));
            auto numEvents = read(m_inotifyFd, events, sizeof(events));
            if (numEvents < 0) {
                perror("read");
                exit(EXIT_FAILURE);
            }
            for (auto i = 0; i < numEvents; ++i) {
                auto event = events[i];
                for (auto& watcher: m_drives[port].watchers) {
                    watcher.onEvent(event);
                }
            }
        }
    }

    udev* m_udev{};
    udev_monitor* m_monitor{};
    int m_inotifyFd{};

    struct DriveUtility {
        path mountPath;
        list<SmartWatcher> watchers;
        thread* eventHandler{};
        bool stopHandler{};
    };
    map<string, DriveUtility> m_drives;
};

int main() {
    cin.tie(nullptr);
    ios::sync_with_stdio(false);

    UsbDeviceManager manager;
//    async(&UsbDeviceManager::observe, &manager);
    thread observer{&UsbDeviceManager::observe, &manager};
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
                if (manager.isSafeToUnmount(port))
                    manager.unmount(port);
                else
                    cout << "Unsafe to unmount" << endl;
            } else {
                manager.unmount(port);
            }
        } else if (option == "q") {
            exit(EXIT_SUCCESS);
        }
    }
}
