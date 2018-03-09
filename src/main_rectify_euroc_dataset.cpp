//
// Created by jg on 18-1-11.
//
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
    img = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    cv::remap(img, imgrec, map1, map2, CV_INTER_LINEAR);
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

//  readCamSettings("/home/jg/Desktop/dso_my_workspace/configs/EuRoC/cam0.yaml", T_BC0, K0, D0);

  int width = 752, height = 480;

  // cv::FileStorage failed to read parameters from config files.
  // input data
  cv::Mat K_l, K_r, R_l, R_r, P_l, P_r, D_l, D_r;
  K_l = (cv::Mat_<float>(3, 3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
  K_r = (cv::Mat_<float>(3, 3) << 457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1.0);
  R_l = (cv::Mat_<float>(3, 3) << 0.999966347530033, -0.001422739138722922, 0.008079580483432283, 0.001365741834644127, 0.9999741760894847, 0.007055629199258132, -0.008089410156878961, -0.007044357138835809, 0.9999424675829176);
  R_r = (cv::Mat_<float>(3, 3) << 0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 0.003680398547259526, 0.9999684752771629, -0.007035845251224894, -0.007729688520722713, 0.007064130529506649, 0.999945173484644);
  P_l = (cv::Mat_<float>(3, 4) << 435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0);
  P_r = (cv::Mat_<float>(3, 4) << 435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0);
  D_l = (cv::Mat_<float>(1, 4) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
  D_r = (cv::Mat_<float>(1, 4) << -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05);
  cv::Mat M1l, M1r, M2l, M2r;

  cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(width, height),CV_32F,M1l,M2l);
  cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(width, height),CV_32F,M1r,M2r);

  vector<string> vec_leftfiles, vec_rightfiles;

  readImageFilePaths(basedir, vec_leftfiles, vec_rightfiles);

  rectifyImages(basedir + "/cam0/data_rec/", vec_leftfiles, M1l, M2l);
  rectifyImages(basedir + "/cam1/data_rec/", vec_rightfiles, M1r, M2r);

//  cv::waitKey(0);

  return 0;
}