#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

template <typename T>
T clamp(const T& value, const T& low, const T& high) {
    return std::max(low, std::min(value, high));
}

double calculateAngleBetweenVectors(const cv::Vec3f& a, const cv::Vec3f& b) {
    float dotProduct = a.dot(b);
    float magnitudeA = cv::norm(a);
    float magnitudeB = cv::norm(b);
    float cosineTheta = dotProduct / (magnitudeA * magnitudeB);
    cosineTheta = clamp(cosineTheta, -1.0f, 1.0f);

    if (cosineTheta < -1.0f || cosineTheta > 1.0f) {
        std::cout << "Invalid cosineTheta: " << cosineTheta << std::endl;
    }



    double angle = std::acos(dotProduct / (magnitudeA * magnitudeB));
    return angle * (180.0 / CV_PI);  // Radians to degrees
}

// y축 회전에 대한 각도 계산 함수
double calculateYRotation(const cv::Vec3f& normalVector) {
    cv::Vec3f projectionOnXYPlane(normalVector[0], normalVector[1], 0);
    double cosineTheta = projectionOnXYPlane.dot(cv::Vec3f(0, 1, 0)) / cv::norm(projectionOnXYPlane);
    return std::acos(cosineTheta) * (180.0 / CV_PI);  // Radians to degrees
}

void detectPlane(const cv::Mat& depthRoi, double& angleBetweenVectors) {
    std::vector<cv::Point3f> points;
    for (int y = 0; y < depthRoi.rows; y++) {
        for (int x = 0; x < depthRoi.cols; x++) {
            ushort depthValue = depthRoi.at<ushort>(y, x);
            if (depthValue > 0) { // Valid depth value
                points.push_back(cv::Point3f(x, y, depthValue));
            }
        }
    }

    // points를 Mat으로 변환
    cv::Mat pointsMat(points.size(), 3, CV_32F);
    for (int i = 0; i < points.size(); i++) {
        pointsMat.at<float>(i, 0) = points[i].x;
        pointsMat.at<float>(i, 1) = points[i].y;
        pointsMat.at<float>(i, 2) = points[i].z;
    }

    // PCA
    cv::PCA pca(pointsMat, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Mat eigenVectors = pca.eigenvectors;
    cv::Vec3f normalVector = eigenVectors.row(eigenVectors.rows - 1);

    if (cv::norm(normalVector) < 1e-5) {
        std::cout << "Invalid normal vector" << std::endl;
        return;
    }

    // 카메라의 벡터 (0, 0, -1)와 평면의 법선 벡터 사이의 각도 계산
    cv::Vec3f cameraVector(0, 0, -1);
    angleBetweenVectors = calculateAngleBetweenVectors(normalVector, cameraVector);
}

int main() {
  rs2::colorizer color_map;
  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

  try
  {
      pipe.start(cfg);
  }
  catch (const rs2::error &e)
  {
      std::cerr << "Failed to open the RealSense camera: " << e.what() << std::endl;
      return -1;
  }

  const auto window_name = "Realsense Depth Frame";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  const auto window_name_color = "Realsense Color Frame";
  cv::namedWindow(window_name_color, cv::WINDOW_AUTOSIZE);

  rs2::align align_to(RS2_STREAM_COLOR);
  rs2::spatial_filter spatial;
  rs2::temporal_filter temporal;
  rs2::hole_filling_filter hole_filling;

  while (true)
  {
      rs2::frameset data = pipe.wait_for_frames();
      data = align_to.process(data);

      rs2::depth_frame depth_frame = data.get_depth_frame();

      // depth_frame = spatial.process(depth_frame);
      // depth_frame = temporal.process(depth_frame);
      depth_frame = hole_filling.process(depth_frame);

      rs2::frame depth = depth_frame;
      rs2::frame color = data.get_color_frame();

      float depth_scale = pipe.get_active_profile().get_device().first<rs2::depth_sensor>().get_depth_scale();
      rs2_intrinsics intrinsics = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

      const int w = depth.as<rs2::video_frame>().get_width();
      const int h = depth.as<rs2::video_frame>().get_height();

      cv::Mat colorMat(cv::Size(w, h), CV_8UC3, (void *)color.get_data(), cv::Mat::AUTO_STEP);
      cv::Mat depthMat(cv::Size(w, h), CV_8UC3, (void *)depth.apply_filter(color_map).get_data(), cv::Mat::AUTO_STEP);
      cv::Mat depth_dist(cv::Size(w, h), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);

      cv::Rect roi(200, 100, 240, 200);
      cv::Mat roiMat = depthMat(roi);

      double angle;
      detectPlane(roiMat, angle);

      // y축 회전 각도 계산
      double yRotationAngle = calculateYRotation(cv::Vec3f(0, 0, -1));  // Assuming the normal vector of the detected plane is (0, 0, -1)

      // 사각형 및 각도 텍스트 추가
      cv::rectangle(colorMat, roi, cv::Scalar(0, 255, 0), 2);
      std::stringstream angleText;
      angleText << "Y Rotation: " << std::fixed << std::setprecision(2) << yRotationAngle << " degrees";
      cv::putText(colorMat, angleText.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

      cv::imshow("Color with ROI and Y Rotation Angle", colorMat);
      cv::imshow("Depth", depthMat);
      if (cv::waitKey(1) >= 0) break;
  }

  return 0;
}
