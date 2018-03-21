//
// Created by jg on 18-3-13.
//

#ifndef DSO_IMURESIDUALS_HPP
#define DSO_IMURESIDUALS_HPP

#include "util/globalCalib.h"
#include "vector"

#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso {
  class FrameHessian;

  class EFIMUResidual;

  class IMUResidual {

    /// \brief The type of the covariance.
    typedef Eigen::Matrix<double, 15, 15> covariance_t;

    /// \brief The type of the information (same matrix dimension as covariance).
    typedef covariance_t information_t;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUResidual(const IMUParameters &imuParameters, const std::vector<IMUMeasurement> &imuData, FrameHessian *from,
                FrameHessian *to);

    ~IMUResidual();

    int redoPreintegration(const SE3 &T_WS, SpeedAndBiases &speedAndBiases) const;

    double linearize(IMUParameters *imuParameters);

    void applyRes(bool copyJacobians);


    EFIMUResidual *efResidual;

    IMURawResidualJacobian *J;
    std::vector<IMUMeasurement> imuData_;
    FrameHessian *from_;
    FrameHessian *to_;
    double t0_;
    double t1_;
    double state_energy;
    double state_NewEnergy;

    IMUParameters imuParameters_;

    mutable information_t information_;
    mutable information_t squareRootInformation_;

    mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
    mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();
    mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero();
    mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();
    mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero();

    mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero();

    mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
    mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();
    mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();

    mutable Eigen::Matrix<double, 15, 15> P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

    mutable SpeedAndBiases speedAndBiases_ref_ = SpeedAndBiases::Zero();
    bool redo_;
    bool redoCounter_ = 0;
  };
}


#endif //DSO_IMURESIDUALS_HPP
