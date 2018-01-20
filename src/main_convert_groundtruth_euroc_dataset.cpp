//
// Created by jg on 18-1-20.
//

#include <fstream>
#include <string>
#include <cstdio>
#include <vector>

struct GTData {
  long long timestampL;
  double timestampD;
  float position[3];
  float quaternion[4];
  float velocity[3];
  float bgyro[3];
  float bacc[3];
};

bool readGT(const std::string &gtFile, std::vector<GTData> &gtDatas) {
  size_t num = 0;

  GTData gt;

  std::ifstream f(gtFile);
  if (!f.is_open()) return false;
  std::string l;
  while (std::getline(f, l)) num++;

  f.clear();
  f.seekg(0, std::ios::beg);

  gtDatas.reserve(num - 1);

  while (std::getline(f, l))
    if (std::sscanf(l.c_str(), "%lld,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                    &gt.timestampL, &gt.position[0], &gt.position[1], &gt.position[2],
                    &gt.quaternion[0], &gt.quaternion[1], &gt.quaternion[2], &gt.quaternion[3],
                    &gt.velocity[0], &gt.velocity[1], &gt.velocity[2],
                    &gt.bgyro[0], &gt.bgyro[1], &gt.bgyro[2],
                    &gt.bacc[0], &gt.bacc[1], &gt.bacc[2])) {
      gt.timestampD = gt.timestampL * 1.0e-9;
      gtDatas.push_back(gt);
    }
  f.close();

  return true;
}

bool outputForEval(const std::string &outputFile, const std::vector<GTData> &gtDatas) {
  std::ofstream f(outputFile);
  if (!f.is_open()) return false;
  std::string l;
  char buf[1000];

  for (const GTData &gtData : gtDatas) {
    std::sprintf(buf, "%lf %f %f %f %f %f %f %f",
                 gtData.timestampD,
                 gtData.position[0], gtData.position[1], gtData.position[2],
                 gtData.quaternion[1], gtData.quaternion[2], gtData.quaternion[3], gtData.quaternion[0]);
    f << buf << std::endl;
  }
  f.close();

  return true;
}

int main(int argc, char **argv) {
  if (argc != 3) return 1;

  std::string gtFile(argv[1]);
  std::string outFile(argv[2]);
  std::vector<GTData> gtDatas;
  readGT(gtFile, gtDatas);
  outputForEval(outFile, gtDatas);

}