//
// Created by jg on 18-4-23.
//

#include <fstream>
#include <string>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

bool readPosesQ(const std::string &file, std::vector<Eigen::Quaterniond> &vec_q,
                std::vector<Eigen::Matrix<double, 3, 1> > &vec_t) {
  size_t poseNum = 0;
  Eigen::Matrix<double, 3, 1> t;
  Eigen::Quaterniond q;
  double time;
  double qx, qy, qz, qw;
  double x, y, z;

  std::ifstream f(file);
  if (!f.is_open()) return false;
  std::string l;
  while (std::getline(f, l)) poseNum++;

  f.clear();
  f.seekg(0, std::ios::beg);

  vec_q.reserve(poseNum);
  vec_t.reserve(poseNum);

  std::printf("poseNum: %ld\n", poseNum);

  while (std::getline(f, l))
    if (std::sscanf(l.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                    &time, &t(0, 0), &t(1, 0), &t(2, 0), &q.x(), &q.y(), &q.z(), &q.w()) == 8) {
      vec_t.push_back(t);
      q.normalize();
      vec_q.push_back(q);
    }
  f.close();

  return true;
}

bool outputPosesT(const std::string &file,
                  const std::vector<Eigen::Quaterniond> &vec_q,
                  const std::vector<Eigen::Matrix<double, 3, 1> > &vec_t) {
  std::ofstream of(file);
  Eigen::Matrix<double, 3, 4> P;
  auto it = vec_q.begin();
  auto it_t = vec_t.begin();
  char l[1024];
  int count = 0;
  for (; it != vec_q.end() && it_t != vec_t.end(); it++, it_t++) {
    P.topLeftCorner<3, 3>() = (*it).toRotationMatrix();
    P.topRightCorner<3, 1>() = (*it_t);
    std::sprintf(l, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                 P(0, 0), P(0, 1), P(0, 2), P(0, 3),
                 P(1, 0), P(1, 1), P(1, 2), P(1, 3),
                 P(2, 0), P(2, 1), P(2, 2), P(2, 3));
    of << l << std::endl;
    count++;
  }

  of.close();
}

bool readPosesT(const std::string &file, std::vector<Eigen::Matrix<double, 3, 4>> &vec_P) {
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

bool outputPosesQ(const std::string &file,
                  const std::vector<Eigen::Matrix<double, 3, 4>> &vec_P) {
  std::ofstream of(file);
  auto it = vec_P.begin();
  for (; it != vec_P.end(); it++) {
    Eigen::Quaterniond q((*it).topLeftCorner<3, 3>());
    of << (*it)(0, 3) << " " << (*it)(1, 3) << " " << (*it)(2, 3) << " " << q.x() << " " << q.y()
       << " " << q.z() << " " << q.w() << "\n";
  }

  of.close();
}

int main(int argc, char **argv) {

  if (argc < 3) return 1;

  std::string inputfile(argv[1]);
  std::string outputfile(argv[2]);

  std::vector<Eigen::Matrix<double, 3, 4> > vec_P;

  std::vector<Eigen::Matrix<double, 3, 1> > vec_t;
  std::vector<Eigen::Quaterniond> vec_q;


//  readPosesT(inputfile, vec_P);
//  std::cout << vec_P.size() << std::endl;
//  outputPosesQ(outputfile, vec_P);

  readPosesQ(inputfile, vec_q, vec_t);
  std::cout << "read: " << vec_q.size() << std::endl;
  outputPosesT(outputfile, vec_q, vec_t);

  return 0;
}