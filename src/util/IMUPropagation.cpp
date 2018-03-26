//
// Created by jg on 18-1-12.
//
#include "NumType.h"
#include "IMUPropagation.h"
#include "globalCalib.h"
#include "okvis_kinematics/include/okvis/kinematics/operators.hpp"
#include "okvis_kinematics/include/okvis/kinematics/Transformation.hpp"

#include <Eigen/Geometry>
#include <iostream>

namespace dso {
  IMUPropagation::IMUPropagation() {

  }

  IMUPropagation::~IMUPropagation() {

  }

  Sophus::Quaterniond
  IMUPropagation::initializeRollPitchFromMeasurements(const std::vector<IMUMeasurement> &imuMeasurements) {

    Vec3 gravity;
    gravity[0] = 0;
    gravity[1] = 0;
    gravity[2] = -1;

    Vec3 omegaAvg;

    for (IMUMeasurement mes : imuMeasurements) {
      omegaAvg += mes.gyr;
    }
    omegaAvg.normalize();

    double theta = abs(acosh(omegaAvg.dot(gravity)));
    Vec3 a = omegaAvg.cross(gravity);

    a *= sin(theta / 2);
    double cosThetaOver2 = cos(theta / 2);

    // R_WS
    return Sophus::Quaterniond(cosThetaOver2, a[0], a[1], a[2]);
  }

  int IMUPropagation::propagate(const std::vector<IMUMeasurement> &imuMeasurements,
                                SE3 T_WS, SpeedAndBias &speedAndBias,
                                const double &t_start, const double &t_end,
                                covariance_t *covariance, jacobian_t *jacobian) {
    double time = t_start;
    double end = t_end;

    assert(imuMeasurements.front().timestamp <= time);
    if (!(imuMeasurements.back().timestamp >= end))
      return -1;

    // initial condition
    Eigen::Vector3d r_0 = T_WS.translation();
    Eigen::Quaterniond q_WS_0 = T_WS.unit_quaternion();
    Eigen::Matrix3d C_WS_0 = T_WS.rotationMatrix();

    // increments (initialize with identity)
    Eigen::Quaterniond Delta_q(1, 0, 0, 0);
    Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
    Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

    // cross matrix accumulation
    Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

    // sub-Jacobians
    Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

    // the Jacobian of the increment (w/0 biases)
    Eigen::Matrix<double, 15, 15> P_delta = Eigen::Matrix<double, 15, 15>::Zero();

    double Delta_t = 0;
    bool hasStarted = false;
    int i = 0;
    for (std::vector<IMUMeasurement>::const_iterator it = imuMeasurements.begin();
         it != imuMeasurements.end(); ++it) {
      Eigen::Vector3d omega_S_0 = it->gyr;
      Eigen::Vector3d acc_S_0 = it->acc;
      Eigen::Vector3d omega_S_1 = (it + 1)->gyr;
      Eigen::Vector3d acc_S_1 = (it + 1)->acc;

      double nexttime;
      if ((it + 1) == imuMeasurements.end())
        nexttime = t_end;
      else
        nexttime = (it + 1)->timestamp;
      // propagation time interval
      double dt = nexttime - time;

      if (end < nexttime) {
        double interval = nexttime - it->timestamp;
        nexttime = t_end;
        dt = nexttime - time;
        const double r = dt / interval;
        omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= 0.0)
        continue;
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
      }

      if (fabs(acc_S_0[0]) > imuParameters.a_max
          || fabs(acc_S_0[1]) > imuParameters.a_max
          || fabs(acc_S_0[2]) > imuParameters.a_max
          || fabs(acc_S_1[0]) > imuParameters.a_max
          || fabs(acc_S_1[1]) > imuParameters.a_max
          || fabs(acc_S_1[2]) > imuParameters.a_max) {
        sigma_a_c *= 100;
      }

      // orientation
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1) - speedAndBias.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt;
      const double sinc_theta_half = sin(theta_half) / theta_half; // TODO: can be more specific
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
      // rotation matrix integral
      const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
      const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
      const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1) - speedAndBias.segment<3>(6));
      const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
      const Eigen::Vector3d acc_integral_1 = acc_integral + 0.5 * (C + C_1) * acc_S_true * dt;
      // rotation matrix double integral
      C_doubleintegral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
      acc_doubleintegral += acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

      // Jacobian part
      dalpha_db_g += dt * C_1;
      const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross +
                                      okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
      const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
      Eigen::Matrix3d dv_db_g_1 = dv_db_g + 0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
      dp_db_g += dt * dv_db_g + 0.25 * dt * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);

      // covariance propagation
      if (covariance) {
        Eigen::Matrix<double, 15, 15> F_delta = Eigen::Matrix<double, 15, 15>::Identity();
        // transform
        F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
            acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
        F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
        F_delta.block<3, 3>(0, 9) = dt * dv_db_g + 0.25 * dt * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(0, 12) = -C_integral * dt + 0.25 * (C + C_1) * dt * dt;
        F_delta.block<3, 3>(3, 9) = -dt * C_1;
        F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(0.5 * (C + C_1) * acc_S_true * dt);
        F_delta.block<3, 3>(6, 9) = 0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
        F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
        P_delta = F_delta * P_delta * F_delta.transpose();
        // add noise. Note that transformations with rotation matrices can be ignored, since the noise is isotropic.
        //F_tot = F_delta*F_tot;
        const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
        P_delta(3, 3) += sigma2_dalpha;
        P_delta(4, 4) += sigma2_dalpha;
        P_delta(5, 5) += sigma2_dalpha;
        const double sigma2_v = dt * sigma_a_c * imuParameters.sigma_a_c;
        P_delta(6, 6) += sigma2_v;
        P_delta(7, 7) += sigma2_v;
        P_delta(8, 8) += sigma2_v;
        const double sigma2_p = 0.5 * dt * dt * sigma2_v;
        P_delta(0, 0) += sigma2_p;
        P_delta(1, 1) += sigma2_p;
        P_delta(2, 2) += sigma2_p;
        const double sigma2_b_g = dt * imuParameters.sigma_gw_c * imuParameters.sigma_gw_c;
        P_delta(9, 9) += sigma2_b_g;
        P_delta(10, 10) += sigma2_b_g;
        P_delta(11, 11) += sigma2_b_g;
        const double sigma2_b_a = dt * imuParameters.sigma_aw_c * imuParameters.sigma_aw_c;
        P_delta(12, 12) += sigma2_b_a;
        P_delta(13, 13) += sigma2_b_a;
        P_delta(14, 14) += sigma2_b_a;
      }

      // memory shift
      Delta_q = Delta_q_1;
      C_integral = C_integral_1;
      acc_integral = acc_integral_1;
      cross = cross_1;
      dv_db_g = dv_db_g_1;
      time = nexttime;

      ++i;

      if (nexttime == t_end)
        break;
    }
    // actual propagation output:
    const Eigen::Vector3d g_W = imuParameters.g * Eigen::Vector3d(0, 0, 6371009).normalized();
//    T_WS.set(r_0+speedAndBias.head<3>()*Delta_t
//             + C_WS_0*(acc_doubleintegral/*-C_doubleintegral*speedAndBiases.segment<3>(6)*/)
//             - 0.5*g_W*Delta_t*Delta_t,
//             q_WS_0*Delta_q);
    T_WS.setQuaternion(q_WS_0 * Delta_q);
    T_WS.translation() = r_0 + speedAndBias.head<3>() * Delta_t
                         + C_WS_0 * (acc_doubleintegral/*-C_doubleintegral*speedAndBiases.segment<3>(6)*/)
                         - 0.5 * g_W * Delta_t * Delta_t;
    speedAndBias.head<3>() += C_WS_0 * (acc_integral/*-C_integral*speedAndBiases.segment<3>(6)*/) - g_W * Delta_t;

    // assign Jacobian, if requested
    if (jacobian) {
      Eigen::Matrix<double, 15, 15> &F = *jacobian;
      F.setIdentity(); // holds for all states, including d/dalpha, d/db_g, d/db_a
      F.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(C_WS_0 * acc_doubleintegral);
      F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * Delta_t;
      F.block<3, 3>(0, 9) = C_WS_0 * dp_db_g;
      F.block<3, 3>(0, 12) = -C_WS_0 * C_doubleintegral;
      F.block<3, 3>(3, 9) = -C_WS_0 * dalpha_db_g;
      F.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(C_WS_0 * acc_integral);
      F.block<3, 3>(6, 9) = C_WS_0 * dv_db_g;
      F.block<3, 3>(6, 12) = -C_WS_0 * C_integral;
    }

    // overall covariance, if requested
    if (covariance) {
      Eigen::Matrix<double, 15, 15> &P = *covariance;
      // transform from local increments to actual states
      Eigen::Matrix<double, 15, 15> T = Eigen::Matrix<double, 15, 15>::Identity();
      T.topLeftCorner<3, 3>() = C_WS_0;
      T.block<3, 3>(3, 3) = C_WS_0;
      T.block<3, 3>(6, 6) = C_WS_0;
      P = T * P_delta * T.transpose();
    }
    return i;
  }
}