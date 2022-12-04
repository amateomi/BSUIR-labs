#include <fcntl.h>
#include <linux/input.h>
#include <unistd.h>

#include <atomic>
#include <fstream>
#include <thread>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

constexpr auto webcamName = "/sys/class/video4linux/video0/device/interface";
constexpr auto webcamDriver = "/sys/class/video4linux/video0/device/uevent";

constexpr auto devNull = "/dev/null";

void videoHandler()
{
    static atomic<bool> isRecording = false;
    static thread recorder;

    auto recordVideo = []() {
        VideoCapture capture { 0 };
        VideoWriter videoWriter { "video.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            capture.get(CAP_PROP_FPS),
            Size { static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT)) } };
        if (!capture.isOpened()) {
            cerr << "Failed to capture video" << endl;
            exit(EXIT_FAILURE);
        }
        vector<Mat> video {};
        while (isRecording) {
            Mat mat;
            capture >> mat;
            video.push_back(mat);
        }
        for (auto&& item : video)
            videoWriter << item;
    };

    isRecording = !isRecording;
    if (isRecording)
        recorder = thread { recordVideo };
    else
        recorder.join();
}

void photoHandler()
{
    VideoCapture capture { 0 };
    Mat mat;
    capture >> mat;
    imwrite("photo.jpg", mat);
}

void keyboardNotifier(const string& keyboardEvent, const function<void()>& eventHandler)
{
    ifstream keyboard { keyboardEvent, ios::binary };
    while (true) {
        input_event event {};
        keyboard.read(reinterpret_cast<char*>(&event), sizeof(event));
        if (event.type == EV_KEY && event.value == 0) {
            auto code = event.code;
            if (code == KEY_F1)
                break;
            if (code == KEY_F2)
                eventHandler();
        }
    }
}

int main(int argc, char* argv[])
{
    cout << ifstream { webcamName }.rdbuf() << endl
         << ifstream { webcamDriver }.rdbuf() << endl;

    if (argc != 3) {
        cerr << "Options is missing:"
             << "\t-p for Photo" << endl
             << "\t-v for Video" << endl
             << "</dev/input/event*> path to keyboard device "
                "(use `cat /proc/bus/input/devices` to find your keyboard event number)"
             << endl;
        exit(EXIT_FAILURE);
    }

    auto eventHandler = [argv]() {
        string arg { argv[1] };
        if (arg == "-p")
            return photoHandler;
        if (arg == "-v")
            return videoHandler;
        cerr << "\"" << arg << "\""
             << "is invalid option" << endl;
        exit(EXIT_FAILURE);
    };

    auto pid = fork();
    if (pid > 0) {
        // Parent
    } else if (pid == 0) {
        // Child
        int fd = open(devNull, O_WRONLY);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        keyboardNotifier(argv[2], eventHandler);
    } else {
        perror("fork");
        exit(errno);
    }
}
