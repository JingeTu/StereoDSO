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


#include "util/globalCalib.h"
#include "vector"

#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso {
  class PointHessian;

  class FrameHessian;

  class CalibHessian;

  class EFResidual;

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  class EFIMUResidual;
  class SpeedAndBiasHessian;
#endif

  enum ResLocation {
    ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE
  };
  enum ResState {
    IN = 0, OOB, OUTLIER
  };

  struct FullJacRowT {
    Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
  };

  class PointFrameResidual {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EFResidual *efResidual;

    static int instanceCounter;


    ResState state_state;
    double state_energy;
    ResState state_NewState;
    double state_NewEnergy;
    double state_NewEnergyWithOutlier;


    void setState(ResState s) { state_state = s; }


    PointHessian *point;
    FrameHessian *host;
    FrameHessian *target;
    RawResidualJacobian *J;


    bool isNew;
    bool staticStereo; //- indicate if this residual is the static stereo residual instead of temperal stereo residual


    Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
    Vec3f centerProjectedTo;

    ~PointFrameResidual();

    PointFrameResidual();

    PointFrameResidual(PointHessian *point_, FrameHessian *host_, FrameHessian *target_);

    double linearize(CalibHessian *HCalib);

    double linearizeStatic(CalibHessian *HCalib);


    void resetOOB() {
      state_NewEnergy = state_energy = 0;
      state_NewState = ResState::OUTLIER;

      setState(ResState::IN);
    };

    void applyRes(bool copyJacobians);

    void debugPlot();

    void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
  };

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  class IMUResidual {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::vector<IMUMeasurement> imuData;

    EFIMUResidual *efIMUResidual;

    FrameHessian *from_f;
    FrameHessian *to_f;
    SpeedAndBiasHessian* from_sb;
    SpeedAndBiasHessian* to_sb;

    RawIMUResidualJacobian *J;

    IMUResidual(SpeedAndBiasHessian* from_sb_, SpeedAndBiasHessian* to_sb_,
                FrameHessian* from_f_, FrameHessian* to_f_,
                std::vector<IMUMeasurement> &imu_data_);
    ~IMUResidual();

    double linearize(IMUParameters *imuParameters);

    void applyRes(bool copyJacobians);

    double t0_;
    double t1_;
    double state_energy;
    double state_NewEnergy;

    int redoPreintegration(const SE3 &T_WS, SpeedAndBias &speedAndBias, IMUParameters *imuParameters) const;

    /// \brief The type of the covariance.
    typedef Eigen::Matrix<double, 15, 15> covariance_t;

    /// \brief The type of the information (same matrix dimension as covariance).
    typedef covariance_t information_t;

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

    mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();
    bool redo_;
    bool redoCounter_ = 0;
  };
#endif
}

