//
// Created by jg on 18-1-12.
//

#ifndef DSO_IMUPROPAGATION_H
#define DSO_IMUPROPAGATION_H

#include <vector>

#include "util/IMUMeasurement.h"

namespace dso {
  class IMUPropagation {
  public:
    IMUPropagation();

    ~IMUPropagation();

//    static Sophus::Quaterniond initializeRollPitchFromMeasurements(const std::vector<IMUMeasurement> &imuMeasurements);

    static Eigen::Matrix3d initializeRollPitchFromMeasurements(const std::vector<IMUMeasurement> &imuMeasurements);

    static int propagate(const std::vector<IMUMeasurement> &imuMeasurements, SE3 &T_WS, SpeedAndBias &speedAndBias,
                         const double &t_start, const double &t_end, covariance_t *covariance, jacobian_t *jacobian);

  private:
  };

}


#endif //DSO_IMUPROPAGATION_H
