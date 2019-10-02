//
// Created by jg on 18-1-21.
//

#include <fstream>
#include <string>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

bool readPoses(const std::string &file, std::vector<Eigen::Matrix<double, 3, 4>> &vec_P) {
  size_t poseNum = 0;
  Eigen::Matrix<double, 3, 4> P;

  std::ifstream f(file);
  if (!f.is_open()) return false;
  std::string l;
  while (std::getline(f, l)) poseNum++;

  f.clear();
  f.seekg(0, std::ios::beg);

  vec_P.reserve(poseNum);

  std::printf("poseNum: %ld\n", poseNum);

  while (std::getline(f, l))
    if (std::sscanf(l.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &P(0, 0), &P(0, 1), &P(0, 2), &P(0, 3),
                    &P(1, 0), &P(1, 1), &P(1, 2), &P(1, 3),
                    &P(2, 0), &P(2, 1), &P(2, 2), &P(2, 3)) == 12)
      vec_P.push_back(P);
  f.close();

  return true;
}

bool readeTimeStamps(const std::string &file, std::vector<double> &vec_Time) {
  size_t timeNum = 0;
  double time = 0.f;

  std::ifstream f(file);
  if (!f.is_open()) return false;
  std::string l;
  while (std::getline(f, l)) timeNum++;

  f.clear();
  f.seekg(0, std::ios::beg);

  vec_Time.reserve(timeNum);

  std::printf("timeNum: %ld\n", timeNum);

  while (std::getline(f, l))
    if (std::sscanf(l.c_str(), "%lf", &time))
      vec_Time.push_back(time);

  f.close();
}

bool combineAndOutput(const std::string &file, const std::vector<double> &vec_Time,
                      const std::vector<Eigen::Matrix<double, 3, 4>> &vec_P) {
  std::ofstream of(file);
  auto it = vec_P.begin();
  auto itTime = vec_Time.begin();
  for (; it != vec_P.end() && itTime != vec_Time.end(); it++, itTime++) {
    Eigen::Quaterniond q((*it).topLeftCorner<3, 3>());
    of << (*itTime) << " " << (*it)(0, 3) << " " << (*it)(1, 3) << " " << (*it)(2, 3) << " " << q.x() << " " << q.y()
       << " " << q.z() << " " << q.w() << "\n";
  }

  of.close();
}

int main(int argc, char **argv) {

  if (argc < 4) return 1;

  std::string inputfile(argv[1]);
  std::string timefile(argv[2]);
  std::string outputfile(argv[3]);

  std::vector<Eigen::Matrix<double, 3, 4>> vec_P;
  std::vector<double> vec_Time;

  readPoses(inputfile, vec_P);

  readeTimeStamps(timefile, vec_Time);

  combineAndOutput(outputfile, vec_Time, vec_P);

  return 0;
}