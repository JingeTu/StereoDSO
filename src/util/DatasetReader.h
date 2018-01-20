/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "util/IMUMeasurement.h"
#include "IOWrapper/ImageRW.h"

#if HAS_ZIPLIB

#include "zip.h"

#endif

#include <boost/thread.hpp>

using namespace dso;


inline int getdir(std::string dir, std::vector<std::string> &files) {
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir(dir.c_str())) == NULL) {
    return -1;
  }

  while ((dirp = readdir(dp)) != NULL) {
    std::string name = std::string(dirp->d_name);

    if (name != "." && name != "..")
      files.push_back(name);
  }
  closedir(dp);


  std::sort(files.begin(), files.end());

  if (dir.at(dir.length() - 1) != '/') dir = dir + "/";
  for (unsigned int i = 0; i < files.size(); i++) {
    if (files[i].at(0) != '/')
      files[i] = dir + files[i];
  }

  return files.size();
}


struct PrepImageItem {
  int id;
  bool isQueud;
  ImageAndExposure *pt;

  inline PrepImageItem(int _id) {
    id = _id;
    isQueud = false;
    pt = 0;
  }

  inline void release() {
    if (pt != 0) delete pt;
    pt = 0;
  }
};

class IMUFileReader {
public:
  IMUFileReader(std::string imuFilePath) {

    std::cout << imuFilePath << std::endl;

    std::ifstream f(imuFilePath);
    if (!f.is_open()) {
      printf("Open IMU file failed: %s", imuFilePath.c_str());
      return;
    }
    size_t imuCount = 0;

    IMUMeasurement item;
    long long timestampL;

    std::string buf;

    while (!f.eof() && f.good()) {
      std::getline(f, buf);
      if (7 == sscanf(buf.c_str(), "%lld,%lf,%lf,%lf,%lf,%lf,%lf",
                      &timestampL, &item.gyr[0], &item.gyr[1], &item.gyr[2],
                      &item.acc[0], &item.acc[1], &item.acc[2])) {
        imuCount++;
      }
    }

    vec_imu_.reserve(imuCount);

    f.clear();
    f.seekg(0, std::ios::beg);


    while (!f.eof() && f.good()) {
      std::getline(f, buf);
      if (7 == sscanf(buf.c_str(), "%lld,%lf,%lf,%lf,%lf,%lf,%lf",
                      &timestampL, &item.gyr[0], &item.gyr[1], &item.gyr[2],
                      &item.acc[0], &item.acc[1], &item.acc[2])) {
        item.timestamp = timestampL * 1e-9;
        vec_imu_.push_back(item);
      }
    }

    printf("IMUFileReader: got %lld imu measurements in %s!\n", (long long int) vec_imu_.size(), imuFilePath.c_str());
  }

  void getIMUMeasurementsBetween(double start, double end, std::vector<IMUMeasurement> &retImuMeasurements) {

//    printf("getIMUMeasurementsBetween [%lf, %lf]\n", start, end);
    std::vector<IMUMeasurement>::iterator startIt = vec_imu_.end();
    std::vector<IMUMeasurement>::iterator endIt = vec_imu_.end();

    for (auto it = vec_imu_.begin(); it != vec_imu_.end(); ++it) {
      if ((*it).timestamp > start && startIt == vec_imu_.end()) {
//        printf("%lf\n", (*it).timestamp);
        startIt = it;
      }
      if ((*it).timestamp > end && endIt == vec_imu_.end()) {
//        printf("%lf\n", (*it).timestamp);
        endIt = it;
        break;
      }
    }

//		assert(startIt != vec_imu_.end() && endIt != vec_imu_.end());

    if (startIt != vec_imu_.end() && endIt != vec_imu_.end()) {
      retImuMeasurements.insert(retImuMeasurements.begin(), startIt, endIt);
    }
  }

private:
  std::vector<IMUMeasurement> vec_imu_;
};


class ImageFolderReader {
public:
  ImageFolderReader(std::string path, std::string datasetName, std::string timestampFile, std::string calibFile,
                    std::string gammaFile, std::string vignetteFile) {
    this->path = path;
    this->calibfile = calibFile;

    getdir(path, files);

    isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

    undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

    widthOrg = undistort->getOriginalSize()[0];
    heightOrg = undistort->getOriginalSize()[1];
    width = undistort->getSize()[0];
    height = undistort->getSize()[1];


    // load timestamps if possible.
    // this is for EuRoCDataset
    if (!timestampFile.empty()) {
      loadTimestamps(datasetName, timestampFile);
    } else
      loadTimestamps();
    printf("ImageFolderReader: got %d files in %s!\n", (int) files.size(), path.c_str());
    assert (timestamps.size() == files.size());

  }

  ~ImageFolderReader() {
#if HAS_ZIPLIB
    if (ziparchive != 0) zip_close(ziparchive);
    if (databuffer != 0) delete databuffer;
#endif


    delete undistort;
  };

  Eigen::VectorXf getOriginalCalib() {
    return undistort->getOriginalParameter().cast<float>();
  }

  Eigen::Vector2i getOriginalDimensions() {
    return undistort->getOriginalSize();
  }

  void getCalibMono(Eigen::Matrix3f &K, int &w, int &h) {
    K = undistort->getK().cast<float>();
    w = undistort->getSize()[0];
    h = undistort->getSize()[1];
  }

  void setGlobalCalibration() {
    int w_out, h_out;
    Eigen::Matrix3f K;
    getCalibMono(K, w_out, h_out);
    setGlobalCameraCalib(w_out, h_out, K);
    setBaseline();
  }

  void setBaseline() {
    baseline = undistort->baseline;
  }

  int getNumImages() {
    return files.size();
  }

  double getTimestamp(int id) {
    if (timestamps.size() == 0) return id * 0.1f;
    if (id >= (int) timestamps.size()) return 0;
    if (id < 0) return 0;
    return timestamps[id];
  }


  void prepImage(int id, bool as8U = false) {

  }


  MinimalImageB *getImageRaw(int id) {
    return getImageRaw_internal(id, 0);
  }

  ImageAndExposure *getImage(int id, bool forceLoadDirectly = false) {
    return getImage_internal(id, 0);
  }


  inline float *getPhotometricGamma() {
    if (undistort == 0 || undistort->photometricUndist == 0) return 0;
    return undistort->photometricUndist->getG();
  }


  // undistorter. [0] always exists, [1-2] only when MT is enabled.
  Undistort *undistort;
private:


  MinimalImageB *getImageRaw_internal(int id, int unused) {
    if (!isZipped) {
      // CHANGE FOR ZIP FILE
      return IOWrap::readImageBW_8U(files[id]);
    } else {
#if HAS_ZIPLIB
      if (databuffer == 0) databuffer = new char[widthOrg * heightOrg * 6 + 10000];
      zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
      long readbytes = zip_fread(fle, databuffer, (long) widthOrg * heightOrg * 6 + 10000);

      if (readbytes > (long) widthOrg * heightOrg * 6) {
        printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,
               (long) widthOrg * heightOrg * 6 + 10000, files[id].c_str());
        delete[] databuffer;
        databuffer = new char[(long) widthOrg * heightOrg * 30];
        fle = zip_fopen(ziparchive, files[id].c_str(), 0);
        readbytes = zip_fread(fle, databuffer, (long) widthOrg * heightOrg * 30 + 10000);

        if (readbytes > (long) widthOrg * heightOrg * 30) {
          printf("buffer still to small (read %ld/%ld). abort.\n", readbytes, (long) widthOrg * heightOrg * 30 + 10000);
          exit(1);
        }
      }

      return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
      printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
      exit(1);
#endif
    }
  }


  ImageAndExposure *getImage_internal(int id, int unused) {
    MinimalImageB *minimg = getImageRaw_internal(id, 0);
    ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
        minimg,
        (exposures.size() == 0 ? 1.0f : exposures[id]),
        (timestamps.size() == 0 ? 0.0 : timestamps[id]));
    delete minimg;
    return ret2;
  }

  inline void loadTimestamps() {
    std::ifstream tr;
    std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
    tr.open(timesFile.c_str());
    while (!tr.eof() && tr.good()) {
      std::string line;
      char buf[1000];
      tr.getline(buf, 1000);

      int id;
      double stamp;
      float exposure = 0;

      if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure)) {
        timestamps.push_back(stamp);
        exposures.push_back(exposure);
      } else if (2 == sscanf(buf, "%d %lf", &id, &stamp)) {
        timestamps.push_back(stamp);
        exposures.push_back(exposure);
      }
    }
    tr.close();

    // check if exposures are correct, (possibly skip)
    bool exposuresGood = ((int) exposures.size() == (int) getNumImages());
    for (int i = 0; i < (int) exposures.size(); i++) {
      if (exposures[i] == 0) {
        // fix!
        float sum = 0, num = 0;
        if (i > 0 && exposures[i - 1] > 0) {
          sum += exposures[i - 1];
          num++;
        }
        if (i + 1 < (int) exposures.size() && exposures[i + 1] > 0) {
          sum += exposures[i + 1];
          num++;
        }

        if (num > 0)
          exposures[i] = sum / num;
      }

      if (exposures[i] == 0) exposuresGood = false;
    }


    if ((int) getNumImages() != (int) timestamps.size()) {
      printf("set timestamps and exposures to zero!\n");
      exposures.clear();
      timestamps.clear();
    }

    if ((int) getNumImages() != (int) exposures.size() || !exposuresGood) {
      printf("set EXPOSURES to zero!\n");
      exposures.clear();
    }

    printf("got %d images and %d timestamps and %d exposures.!\n", (int) getNumImages(), (int) timestamps.size(),
           (int) exposures.size());
  }

  inline void loadTimestamps(std::string datasetName, std::string timestampFile) {
    if (datasetName == "euroc") {
      std::ifstream tr;
      std::string timesFile = path.substr(0, path.find_last_of('/')) + "/data.csv";
      tr.open(timesFile.c_str());
      while (!tr.eof() && tr.good()) {
        std::string line;
        char buf[1000];
        tr.getline(buf, 1000);

        int id;
        long long stampL;
        double stamp;
        float exposure = 0;
        char filename[1000];

        if (2 == sscanf(buf, "%lld,%s", &stampL, filename)) {
          stamp = stampL * 1e-9;
          timestamps.push_back(stamp);
//				exposures.push_back(exposure);
        }
      }
      tr.close();
    } else if (datasetName == "kitti") {
      std::ifstream tr;
      std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
      tr.open(timesFile.c_str());
      while (!tr.eof() && tr.good()) {
        std::string line;
        char buf[1000];
        tr.getline(buf, 1000);

        int id;
        long long stampL;
        double stamp;
        float exposure = 0;
        char filename[1000];

        if (1 == sscanf(buf, "%lf", &stamp)) {
          timestamps.push_back(stamp);
        }
      }
      tr.close();
    }
    printf("ImageFolder Reader: got %ld timestamps in %s.\n", timestamps.size(), timestampFile.c_str());
  }


  std::vector<ImageAndExposure *> preloadedImages;
  std::vector<std::string> files;
  std::vector<double> timestamps;
  std::vector<float> exposures;

  int width, height;
  int widthOrg, heightOrg;

  std::string path;
  std::string calibfile;

  bool isZipped;

#if HAS_ZIPLIB
  zip_t *ziparchive;
  char *databuffer;
#endif
};

