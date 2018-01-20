//
// Created by jg on 18-1-11.
//

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>

using namespace std;

void readCamSettings(const std::string &settingsfile, cv::Mat &T_BC, cv::Mat &K, cv::Mat D) {
  cv::FileStorage settings;
  try {
    settings.open(settingsfile, cv::FileStorage::READ);
  }
  catch (cv::Exception e) {
    cerr << e.msg << endl;
  }

  double fx, fy, cx, cy;

  cout << settings.isOpened() << endl;

  settings["T_BS"] >> T_BC;
  settings["camera.fu"] >> fx;
  settings["camera.fv"] >> fy;
  settings["camera.cu"] >> cx;
  settings["camera.cv"] >> cy;

  K = cv::Mat::eye(3, 3, CV_32F);

  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
}

bool readImageFilePathsSingle(const std::string &basedir, std::vector<std::string> &vec_files) {
  ifstream f(basedir + "data.csv");
  if (!f.is_open()) return false;
  string l;

  size_t num = 0;
  size_t timestamp;
  char buf[256];

  getline(f, l);

  while (getline(f, l)) num++;
  vec_files.clear();
  vec_files.reserve(num);

  f.clear();
  f.seekg(0, ios::beg);

  getline(f, l);

  while (getline(f, l))
    if (sscanf(l.c_str(), "%ld,%s", &timestamp, buf) == 2)
      vec_files.push_back(string(basedir + "data/" + buf));

  f.close();

  return true;
}

void readImageFilePaths(const std::string &basedir, std::vector<std::string> &vec_leftfiles,
                        std::vector<std::string> &vec_rightfiles) {
  // cam0
  if (!readImageFilePathsSingle(basedir + "/cam0/", vec_leftfiles)) cout << "something wrong." << endl;
  // cam1
  if (!readImageFilePathsSingle(basedir + "/cam1/", vec_rightfiles)) cout << "something wrong." << endl;
}

void rectifyImages(const std::string &outputdir, const std::vector<std::string> &vec_files,
                   const cv::Mat &map1, const cv::Mat &map2) {
  if (vec_files.empty()) return;

  size_t startpos = vec_files[0].find_last_of('/') + 1;

  cout << vec_files[0].substr(vec_files[0].find_last_of('/') + 1, vec_files[0].size()) << endl;

  int r = system(("rm -rf " + outputdir).c_str());
  r = system(("mkdir " + outputdir).c_str());

  cv::Mat img, imgrec;

  for (const string &filename : vec_files) {
    img = cv::imread(filename);
    cv::remap(img, imgrec, map1, map2, CV_INTER_CUBIC);
    cv::imwrite(outputdir + filename.substr(startpos, filename.size()), imgrec);
  }
}

int main(int argc, char **argv) {

  // argv[1] = /home/jg/Documents/Datasets/EuRoC/MH_01_easy/mav0
  string basedir(argv[1]);
//  string cam0file(basedir + "/cam0/sensor.yaml");
//  string cam1file(basedir + "/cam1/sensor.yaml");

  cv::Mat T_BC0, T_BC1;
  cv::Mat K0, K1;
  cv::Mat D0, D1;

  readCamSettings("/home/jg/Desktop/dso_my_workspace/configs/EuRoC/cam0.yaml", T_BC0, K0, D0);

  int width = 752, height = 480;

  // cv::FileStorage failed to read parameters from config files.
  // input data
  T_BC0 = (cv::Mat_<float>(4, 4) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
      0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
      -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
      0.0, 0.0, 0.0, 1.0);
  K0 = (cv::Mat_<float>(3, 3) << 458.654, 0.0, 376.215,
      0.0, 457.296, 248.375,
      0.0, 0.0, 1.0);
  D0 = (cv::Mat_<float>(1, 4) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

  T_BC1 = (cv::Mat_<float>(4, 4) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
      0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
      -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
      0.0, 0.0, 0.0, 1.0);
  K1 = (cv::Mat_<float>(3, 3) << 457.587, 0.0, 379.999,
      0.0, 456.134, 255.238,
      0.0, 0.0, 1.0);
  D1 = (cv::Mat_<float>(1, 4) << -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);

  // get remap
  cv::Mat T_C0C1 = T_BC0.inv() * T_BC1;
  cv::Mat R01 = T_C0C1.colRange(0, 3).rowRange(0, 3);
  cv::Mat t = cv::Mat(T_C0C1.col(3).rowRange(0, 3));
  float b = sqrtf(t.dot(t));
  cv::Mat tp = (cv::Mat_<float>(3, 1) << b, 0.0, 0.0);

  float theta = abs(acosf(t.dot(tp) / (sqrtf(t.dot(t)) * sqrtf(tp.dot(tp)))));
  cv::Mat a = (t / t.dot(t)).cross(tp / tp.dot(tp));
  cv::Mat Rp;
  cv::Rodrigues(a * theta, Rp);

  cv::Mat map1_0, map2_0, map1_1, map2_1;
  cv::initUndistortRectifyMap(K0, D0, Rp, K0, cv::Size(width, height), CV_32F, map1_0, map2_0);
  cv::initUndistortRectifyMap(K1, D1, Rp * R01, K0, cv::Size(width, height), CV_32F, map1_1, map2_1);

  vector<string> vec_leftfiles, vec_rightfiles;

  readImageFilePaths(basedir, vec_leftfiles, vec_rightfiles);

  rectifyImages(basedir + "/cam0/data_rec/", vec_leftfiles, map1_0, map2_0);
  rectifyImages(basedir + "/cam1/data_rec/", vec_rightfiles, map1_1, map2_1);

//  cv::waitKey(0);

  return 0;
}