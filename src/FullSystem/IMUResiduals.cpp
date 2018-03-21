//
// Created by jg on 18-3-13.
//

#include <util/FrameShell.h>
#include "IMUResiduals.hpp"
#include "HessianBlocks.h"

#include "okvis_kinematics/include/okvis/kinematics/operators.hpp"
#include "okvis_kinematics/include/okvis/kinematics/Transformation.hpp"

namespace dso {

  static __inline__ double sinc(double x) {
    if (fabs(x) > 1e-6) {
      return sin(x) / x;
    }
    else {
      static const double c_2 = 1.0 / 6.0;
      static const double c_4 = 1.0 / 120.0;
      static const double c_6 = 1.0 / 5040.0;
      const double x_2 = x * x;
      const double x_4 = x_2 * x_2;
      const double x_6 = x_2 * x_2 * x_2;
      return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
    }
  }

  IMUResidual::IMUResidual(const IMUParameters &imuParameters, const std::vector<IMUMeasurement> &imuData,
                           FrameHessian *from, FrameHessian *to)
      : imuData_(std::move(imuData)), from_(from), to_(to),
        t0_(from_->shell->timestamp), t1_(to_->shell->timestamp) {
    J = new IMURawResidualJacobian();
  }

  IMUResidual::~IMUResidual() {

  }

  int IMUResidual::redoPreintegration(const SE3 &T_WS, SpeedAndBiases &speedAndBiases) const {
    // now the propagation
    double time = t0_;
    double end = t1_;

    // sanity check:
    assert(imuData_.front().timestamp <= time);
    if (!(imuData_.back().timestamp >= end))
      return -1;  // nothing to do...

    // increments (initialise with identity)
    Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
    C_integral_ = Eigen::Matrix3d::Zero();
    C_doubleintegral_ = Eigen::Matrix3d::Zero();
    acc_integral_ = Eigen::Vector3d::Zero();
    acc_doubleintegral_ = Eigen::Vector3d::Zero();

    // cross matrix accumulatrion
    cross_ = Eigen::Matrix3d::Zero();

    // sub-Jacobians
    dalpha_db_g_ = Eigen::Matrix3d::Zero();
    dv_db_g_ = Eigen::Matrix3d::Zero();
    dp_db_g_ = Eigen::Matrix3d::Zero();

    // the Jacobian of the increment (w/o biases)
    P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

    //Eigen::Matrix<double, 15, 15> F_tot;
    //F_tot.setIdentity();

    double Delta_t = 0;
    bool hasStarted = false;
    int i = 0;
    for (std::vector<IMUMeasurement>::const_iterator it = imuData_.begin();
         it != imuData_.end(); ++it) {

      Eigen::Vector3d omega_S_0 = it->gyr;
      Eigen::Vector3d acc_S_0 = it->acc;
      Eigen::Vector3d omega_S_1 = (it + 1)->gyr;
      Eigen::Vector3d acc_S_1 = (it + 1)->acc;

      // time delta
      double nexttime;
      if ((it + 1) == imuData_.end()) {
        nexttime = t1_;
      }
      else
        nexttime = (it + 1)->timestamp;
      double dt = (nexttime - time);

      if (end < nexttime) {
        double interval = (nexttime - it->timestamp);
        nexttime = t1_;
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
      double sigma_g_c = imuParameters_.sigma_g_c;
      double sigma_a_c = imuParameters_.sigma_a_c;

      if (fabs(omega_S_0[0]) > imuParameters_.g_max
          || fabs(omega_S_0[1]) > imuParameters_.g_max
          || fabs(omega_S_0[2]) > imuParameters_.g_max
          || fabs(omega_S_1[0]) > imuParameters_.g_max
          || fabs(omega_S_1[1]) > imuParameters_.g_max
          || fabs(omega_S_1[2]) > imuParameters_.g_max) {
        sigma_g_c *= 100;
        LOG(WARNING) << "gyr saturation";
      }

      if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
          || fabs(acc_S_0[2]) > imuParameters_.a_max
          || fabs(acc_S_1[0]) > imuParameters_.a_max
          || fabs(acc_S_1[1]) > imuParameters_.a_max
          || fabs(acc_S_1[2]) > imuParameters_.a_max) {
        sigma_a_c *= 100;
        LOG(WARNING) << "acc saturation";
      }

      // actual propagation
      // orientation:
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
                                            - speedAndBiases.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt;
      const double sinc_theta_half = sinc(theta_half);
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
      // rotation matrix integral:
      const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
      const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
      const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
                                          - speedAndBiases.segment<3>(6));
      const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
      const Eigen::Vector3d acc_integral_1 = acc_integral_
                                             + 0.5 * (C + C_1) * acc_S_true * dt;
      // rotation matrix double integral:
      C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
      acc_doubleintegral_ += acc_integral_ * dt
                             + 0.25 * (C + C_1) * acc_S_true * dt * dt;

      // Jacobian parts
      dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
      const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
                                      + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
      const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
      Eigen::Matrix3d dv_db_g_1 = dv_db_g_
                                  + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      dp_db_g_ += dt * dv_db_g_
                  + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

      // covariance propagation
      Eigen::Matrix<double, 15, 15> F_delta =
          Eigen::Matrix<double, 15, 15>::Identity();
      // transform
      F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
          acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
      F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
      F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
                                  + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
                                   + 0.25 * (C + C_1) * dt * dt;
      F_delta.block<3, 3>(3, 9) = -dt * C_1;
      F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
          0.5 * (C + C_1) * acc_S_true * dt);
      F_delta.block<3, 3>(6, 9) = 0.5 * dt
                                  * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
      P_delta_ = F_delta * P_delta_ * F_delta.transpose();
      // add noise. Note that transformations with rotation matrices can be ignored, since the noise is isotropic.
      //F_tot = F_delta*F_tot;
      const double sigma2_dalpha = dt * sigma_g_c
                                   * sigma_g_c;
      P_delta_(3, 3) += sigma2_dalpha;
      P_delta_(4, 4) += sigma2_dalpha;
      P_delta_(5, 5) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * sigma_a_c;
      P_delta_(6, 6) += sigma2_v;
      P_delta_(7, 7) += sigma2_v;
      P_delta_(8, 8) += sigma2_v;
      const double sigma2_p = 0.5 * dt * dt * sigma2_v;
      P_delta_(0, 0) += sigma2_p;
      P_delta_(1, 1) += sigma2_p;
      P_delta_(2, 2) += sigma2_p;
      const double sigma2_b_g = dt * imuParameters_.sigma_gw_c * imuParameters_.sigma_gw_c;
      P_delta_(9, 9) += sigma2_b_g;
      P_delta_(10, 10) += sigma2_b_g;
      P_delta_(11, 11) += sigma2_b_g;
      const double sigma2_b_a = dt * imuParameters_.sigma_aw_c * imuParameters_.sigma_aw_c;
      P_delta_(12, 12) += sigma2_b_a;
      P_delta_(13, 13) += sigma2_b_a;
      P_delta_(14, 14) += sigma2_b_a;

      // memory shift
      Delta_q_ = Delta_q_1;
      C_integral_ = C_integral_1;
      acc_integral_ = acc_integral_1;
      cross_ = cross_1;
      dv_db_g_ = dv_db_g_1;
      time = nexttime;

      ++i;

      if (nexttime == t1_)
        break;

    }

    // store the reference (linearisation) point
    speedAndBiases_ref_ = speedAndBiases;

    // get the weighting:
    // enforce symmetric
    P_delta_ = 0.5 * P_delta_ + 0.5 * P_delta_.transpose().eval();

    // calculate inverse
    information_ = P_delta_.inverse();
    information_ = 0.5 * information_ + 0.5 * information_.transpose().eval();

    // square root
    Eigen::LLT<information_t> lltOfInformation(information_);
    squareRootInformation_ = lltOfInformation.matrixL().transpose();

    return i;
  }

  double IMUResidual::linearize(IMUParameters *imuParameters) {
    SE3 T_WS_0 = (T_SC0 * from_->get_worldToCam_evalPT()).inverse();
    SE3 T_WS_1 = (T_SC0 * to_->get_worldToCam_evalPT()).inverse();

    // get speed and bias
    SpeedAndBiases speedAndBiases_0 = from_->shell->speedAndBiases;
    SpeedAndBiases speedAndBiases_1 = to_->shell->speedAndBiases;

    // this will NOT be changed:
    const Eigen::Matrix3d C_WS_0 = T_WS_0.rotationMatrix();
    const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

    // call the propagation
    const double Delta_t = (t1_ - t0_);
    Eigen::Matrix<double, 6, 1> Delta_b = speedAndBiases_0.tail<6>() - speedAndBiases_ref_.tail<6>();

    redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
    if (redo_) {
      redoPreintegration(T_WS_0, speedAndBiases_0);
      redoCounter_++;
      Delta_b.setZero();
      redo_ = false;
    }

    {
      const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

      // assign Jacobian w.r.t. x0
      Eigen::Matrix<double, 15, 15> F0 =
          Eigen::Matrix<double, 15, 15>::Identity(); // holds for d/db_g, d/db_a
      const Eigen::Vector3d delta_p_est_W =
          T_WS_0.translation() - T_WS_1.translation() + speedAndBiases_0.head<3>() * Delta_t -
          0.5 * g_W * Delta_t * Delta_t;
      const Eigen::Vector3d delta_v_est_W =
          speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W * Delta_t;
      const Eigen::Quaterniond Dq = okvis::kinematics::deltaQ(-dalpha_db_g_ * Delta_b.head<3>()) * Delta_q_;
      F0.block<3, 3>(0, 0) = C_S0_W;
      F0.block<3, 3>(0, 3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
      F0.block<3, 3>(0, 6) = C_S0_W * Eigen::Matrix3d::Identity() * Delta_t;
      F0.block<3, 3>(0, 9) = dp_db_g_;
      F0.block<3, 3>(0, 12) = -C_doubleintegral_;
      F0.block<3, 3>(3, 3) = (okvis::kinematics::plus(Dq * T_WS_1.unit_quaternion().inverse()) *
                              okvis::kinematics::oplus(T_WS_0.unit_quaternion())).topLeftCorner<3, 3>();
      F0.block<3, 3>(3, 9) = (okvis::kinematics::oplus(T_WS_1.unit_quaternion().inverse() * T_WS_0.unit_quaternion()) *
                              okvis::kinematics::oplus(Dq)).topLeftCorner<3, 3>() * (-dalpha_db_g_);
      F0.block<3, 3>(6, 3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
      F0.block<3, 3>(6, 6) = C_S0_W;
      F0.block<3, 3>(6, 9) = dv_db_g_;
      F0.block<3, 3>(6, 12) = -C_integral_;

      // assign Jacobian w.r.t. x1
      Eigen::Matrix<double, 15, 15> F1 =
          -Eigen::Matrix<double, 15, 15>::Identity(); // holds for the biases
      F1.block<3, 3>(0, 0) = -C_S0_W;
      F1.block<3, 3>(3, 3) = -(okvis::kinematics::plus(Dq) *
                               okvis::kinematics::oplus(T_WS_0.unit_quaternion()) *
                               okvis::kinematics::plus(T_WS_1.unit_quaternion().inverse())).topLeftCorner<3, 3>();
      F1.block<3, 3>(6, 6) = -C_S0_W;

      // the overall error vector
      Eigen::Matrix<double, 15, 1> error;
      error.segment<3>(0) = C_S0_W * delta_p_est_W + acc_doubleintegral_ + F0.block<3, 6>(0, 9) * Delta_b;
      error.segment<3>(3) = 2 * (Dq * (T_WS_1.unit_quaternion().inverse() *
                                       T_WS_0.unit_quaternion())).vec(); //2*T_WS_0.q()*Dq*T_WS_1.q().inverse();//
      error.segment<3>(6) = C_S0_W * delta_v_est_W + acc_integral_ + F0.block<3, 6>(6, 9) * Delta_b;
      error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();

      // error weighting
      J->resF = squareRootInformation_ * error;

      // get the Jacobians
      {
        Eigen::Matrix<double, 15, 6> J0_minimal = squareRootInformation_ * F0.block<15, 6>(0, 0);

      }
      J->Jxdxi[0] = squareRootInformation_ * F0.block<15, 6>(0, 0);
      J->Jxdsb[0] = squareRootInformation_ * F0.block<15, 9>(0, 6);
      J->Jxdxi[1] = squareRootInformation_ * F1.block<15, 6>(0, 0);
      J->Jxdsb[1] = squareRootInformation_ * F1.block<15, 9>(0, 6);
    }
    state_NewEnergy = J->resF.norm();
    return state_NewEnergy;
  }

  void IMUResidual::applyRes(bool copyJacobians) {
    if (copyJacobians)
      efResidual->takeDataF();

    efResidual->isActiveAndIsGoodNEW = true;
    state_energy = state_NewEnergy;
  }
}