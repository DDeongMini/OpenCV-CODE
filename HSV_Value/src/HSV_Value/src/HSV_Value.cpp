#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat frame;
Mat hsv_frame;

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Vec3b hsv_value = hsv_frame.at<Vec3b>(y, x);
        cout << "HSV value at (" << x << ", " << y << "): ["
             << (int)hsv_value[0] << ", "  // H value
             << (int)hsv_value[1] << ", "  // S value
             << (int)hsv_value[2] << "]"   // V value
             << endl;
    }
}

int main() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cerr << "Error! Unable to open the webcam." << endl;
        return -1;
    }

    namedWindow("Webcam Frame", 1);
    setMouseCallback("Webcam Frame", mouseCallback);

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            cerr << "Error! Blank frame grabbed." << endl;
            break;
        }

        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        imshow("Webcam Frame", frame);

        if (waitKey(10) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
