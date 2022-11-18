#include <sys/io.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace std;

const uint32_t pciEnableBit = 0x80000000;
const uint32_t pciConfigAddress = 0xcf8;
const uint32_t pciConfigData = 0xcfc;

const auto pciNamesDatabase = "/usr/share/hwdata/pci.ids";
const auto stopLine = "# List of known device classes, subclasses and programming interfaces";

using vendorId = uint16_t;
using deviceId = uint16_t;

using vendorName = string;
using deviceName = string;

using vendorDeviceId = set<pair<vendorId, deviceId>>;

using devicesInfo = map<deviceId, deviceName>;
using vendorDevices = map<vendorId, pair<vendorName, devicesInfo>>;

vendorDeviceId grubPciIds()
{
    vendorDeviceId vendorDeviceId;
    for (uint8_t bus = 0; bus != UINT8_MAX; bus++) {
        for (uint8_t dev = 0; dev < 32; dev++) {
            for (uint8_t func = 0; func < 8; func++) {
                outl(pciEnableBit | (bus << 16) | (dev << 11) | (func << 8), pciConfigAddress);
                const uint32_t data = inl(pciConfigData);
                if (data == 0xFFFFFFFF) {
                    continue;
                }
                const uint16_t vendorId = data;
                const uint16_t deviceId = data >> 16;
                vendorDeviceId.emplace(vendorId, deviceId);
            }
        }
    }
    return vendorDeviceId;
}

vendorDevices parseDatabase()
{
    vendorDevices devices;

    ifstream input { pciNamesDatabase };
    string line;
    int vendorId;
    devicesInfo devicesInfo;

    for (getline(input, line); line != stopLine; getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        if (line[0] == '\t' && line[1] == '\t')
            continue;
        if (line[0] != '\t') {
            if (!devices.empty()) {
                devices[vendorId].second = devicesInfo;
                devicesInfo.clear();
            }
            vendorId = stoi(line, nullptr, 16);
            devices[vendorId].first = line.substr(line.find("  ") + 2);
        } else if (line[0] == '\t') {
            const auto deviceId = stoi(line, nullptr, 16);
            devicesInfo[deviceId] = line.substr(line.find("  ") + 2);
        }
    }
    return devices;
}

int main()
{
    if (iopl(3)) {
        perror("Root privileges required");
        exit(EXIT_FAILURE);
    }

    auto ids = grubPciIds();
    auto db = parseDatabase();

    for (const auto& [vendorId, deviceId] : ids) {
        cout << "vendor name: " << db[vendorId].first << '\n'
             << "device name: " << db[vendorId].second[deviceId] << '\n'
             << "vendor id: " << hex << vendorId << '\n'
             << "device id: " << hex << deviceId << '\n'
             << endl;
    }
}
