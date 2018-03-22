//
// Created by jg on 18-1-12.
//

#ifndef DSO_IMUMEASUREMENT_H
#define DSO_IMUMEASUREMENT_H

#include <Eigen/Dense>
#include "NumType.h"

namespace dso {
  struct IMUMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Vector3d gyr;
    Eigen::Vector3d acc;
  };

  struct IMUParameters {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE3 T_BS; ///< Transformation from Body frame to IMU (sensor frame S).
    double a_max;  ///< Accelerometer saturation. [m/s^2]
    double g_max;  ///< Gyroscope saturation. [rad/s]
    double sigma_g_c;  ///< Gyroscope noise density.
    double sigma_bg;  ///< Initial gyroscope bias.
    double sigma_a_c;  ///< Accelerometer noise density.
    double sigma_ba;  ///< Initial accelerometer bias
    double sigma_gw_c; ///< Gyroscope drift noise density.
    double sigma_aw_c; ///< Accelerometer drift noise density.
    double tau;  ///< Reversion time constant of accerometer bias. [s]
    double g;  ///< Earth acceleration.
    Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
    int rate;  ///< IMU rate in Hz.
  };
}

#endif //DSO_IMUMEASUREMENT_H
