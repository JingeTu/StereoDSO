//
// Created by jg on 18-1-12.
//
#include "NumType.h"
#include "IMUPropagation.h"
#include "globalCalib.h"
#include "okvis_kinematics/include/okvis/kinematics/operators.hpp"
#include "okvis_kinematics/include/okvis/kinematics/Transformation.hpp"
#include "globalFuncs.h"
#include <Eigen/Geometry>
#include <iostream>

namespace dso {
  IMUPropagation::IMUPropagation() {

  }

  IMUPropagation::~IMUPropagation() {

  }

//  Sophus::Quaterniond
//  IMUPropagation::initializeRollPitchFromMeasurements(const std::vector<IMUMeasurement> &imuMeasurements) {
//
//    Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
//    for (IMUMeasurement mes : imuMeasurements) {
//      acc_B += mes.acc;
//    }
//    acc_B /= double(imuMeasurements.size());
//    Eigen::Vector3d e_acc = acc_B.normalized();
//
//    Eigen::Vector3d ez_W(0.0, 0.0, -1.0);
//    Eigen::Matrix<double, 6, 1> poseIncrement;
//    poseIncrement.head<3>() = Eigen::Vector3d::Zero();
//    poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
//    double angle = std::acos(ez_W.transpose() * e_acc);
//    poseIncrement.tail<3>() *= angle;
//    poseIncrement *= -1;
//
//    Eigen::Vector4d dq;
//    double halfnorm = 0.5 * poseIncrement.template tail<3>().norm();
//    dq.template head<3>() = sinc(halfnorm) * 0.5 * poseIncrement.template tail<3>();
//    dq[3] = cos(halfnorm);
//
//    return Sophus::Quaterniond(dq);
//  }

  Eigen::Matrix3d
  IMUPropagation::initializeRollPitchFromMeasurements(const std::vector<IMUMeasurement> &imuMeasurements) {
    Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
    for (IMUMeasurement mes : imuMeasurements) {
      acc_B += mes.acc;
    }
    acc_B /= double(imuMeasurements.size());
    Eigen::Vector3d a = acc_B.normalized();

    Eigen::Vector3d g_W(0.0, 1.0, 0.0);
    Eigen::Vector3d b = g_W.cross(a).normalized();
    double theta = std::acos(a.transpose() * g_W);
    Eigen::AngleAxisd aa(theta, b);

    Eigen::Matrix3d Rsw = aa.toRotationMatrix();
    return Rsw;
  }

  int IMUPropagation::propagate(const std::vector<IMUMeasurement> &imuData,
                                SE3 &T_WS, SpeedAndBias &speedAndBias,
                                const double &t_start, const double &t_end,
                                covariance_t *covariance, jacobian_t *jacobian) {
    // now the propagation
    double time = t_start;
    double end = t_end;

    // sanity check:
    assert(imuData.front().timestamp <= time);
    if (!(imuData.back().timestamp >= end))
      return -1;  // nothing to do...

    // increments (initialise with identity)
    Eigen::Matrix3d Delta_tilde_R_ij_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Delta_tilde_v_ij_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d Delta_tilde_p_ij_ = Eigen::Vector3d::Zero();

    double Delta_t = 0;
    bool hasStarted = false;
    int i = 0;
    for (std::vector<IMUMeasurement>::const_iterator it = imuData.begin();
         it != imuData.end(); ++it) {

      Eigen::Vector3d omega_S_0 = it->gyr;
      Eigen::Vector3d acc_S_0 = it->acc;
      Eigen::Vector3d omega_S_1 = (it + 1)->gyr;
      Eigen::Vector3d acc_S_1 = (it + 1)->acc;

      // time delta
      double nexttime;
      if ((it + 1) == imuData.end()) {
        nexttime = t_end;
      }
      else
        nexttime = (it + 1)->timestamp;
      double dt = (nexttime - time);

      if (end < nexttime) {
        double interval = (nexttime - it->timestamp);
        nexttime = t_end;
        dt = (nexttime - time);
        const double r = dt / interval;
        omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= 0.0) {
        continue;
      }
      Delta_t += dt;

      if (!hasStarted) {
        hasStarted = true;
        const double r = dt / (nexttime - it->timestamp);
        omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
      }

      // ensure integrity
      double sigma_g_c = imuParameters.sigma_g_c;
      double sigma_a_c = imuParameters.sigma_a_c;

      if (fabs(omega_S_0[0]) > imuParameters.g_max
          || fabs(omega_S_0[1]) > imuParameters.g_max
          || fabs(omega_S_0[2]) > imuParameters.g_max
          || fabs(omega_S_1[0]) > imuParameters.g_max
          || fabs(omega_S_1[1]) > imuParameters.g_max
          || fabs(omega_S_1[2]) > imuParameters.g_max) {
        sigma_g_c *= 100;
        LOG(WARNING) << "gyr saturation";
      }

      if (fabs(acc_S_0[0]) > imuParameters.a_max || fabs(acc_S_0[1]) > imuParameters.a_max
          || fabs(acc_S_0[2]) > imuParameters.a_max
          || fabs(acc_S_1[0]) > imuParameters.a_max
          || fabs(acc_S_1[1]) > imuParameters.a_max
          || fabs(acc_S_1[2]) > imuParameters.a_max) {
        sigma_a_c *= 100;
        LOG(WARNING) << "acc saturation";
      }

      // actual propagation (A.10)
      // R:
      const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1) - speedAndBias.segment<3>(3));
      Eigen::Matrix3d Delta_R = SO3::exp(
          omega_S_true * dt).matrix();//SO3::exp(okvis::kinematics::crossMx(omega_S_true * dt));
      Eigen::Matrix3d Delta_tilde_R_ij = Delta_tilde_R_ij_ * Delta_R;
      // v:
      const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1) - speedAndBias.segment<3>(6));
      Eigen::Vector3d Delta_tilde_v_ij = Delta_tilde_v_ij_ + Delta_tilde_R_ij_ * acc_S_true * dt;
      // p:
      Eigen::Vector3d Delta_tilde_p_ij = Delta_tilde_p_ij_ + 1.5 * Delta_tilde_R_ij_ * acc_S_true * dt * dt;

      // memory shift
      Delta_tilde_R_ij_ = Delta_tilde_R_ij;
      Delta_tilde_v_ij_ = Delta_tilde_v_ij;
      Delta_tilde_p_ij_ = Delta_tilde_p_ij;
      time = nexttime;

      ++i;

      if (nexttime == t_end)
        break;

    }

    const Eigen::Vector3d g_W = imuParameters.g * Eigen::Vector3d(0, -1, 0).normalized();
    Eigen::Matrix3d C_WS = T_WS.rotationMatrix();
    Eigen::Vector3d r = T_WS.translation();


    T_WS.setRotationMatrix(C_WS * Delta_tilde_R_ij_);
    T_WS.translation() = C_WS * Delta_tilde_p_ij_
                         + r
                         + speedAndBias.head<3>() * Delta_t
                         + 0.5 * g_W * Delta_t * Delta_t;
    speedAndBias.head<3>() = C_WS * Delta_tilde_v_ij_ + speedAndBias.head<3>() + g_W * Delta_t;

    return i;
  }
}