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


#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"
#include "util/settings.h"
#include <boost/thread.hpp>

namespace dso {

  class PointFrameResidual;

  class CalibHessian;

  class FrameHessian;

  class PointHessian;

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  class SpeedAndBiasHessian;

  class EFSpeedAndBias;

  class IMUResidual;
#endif

  class EFResidual;

  class EFPoint;

  class EFFrame;

  class EnergyFunctional;

  class EFResidual {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame *host_, EFFrame *target_) :
        data(org), point(point_), host(host_), target(target_) {
      isLinearized = false;
      isActiveAndIsGoodNEW = false;
      J = new RawResidualJacobian();
      assert(((long) this) % 16 == 0);
      assert(((long) J) % 16 == 0);
    }

    inline ~EFResidual() {
      delete J;
    }


    void takeDataF();


    void fixLinearizationF(EnergyFunctional *ef);

    // structural pointers
    PointFrameResidual *data;
    int hostIDX, targetIDX;
    EFPoint *point;
    EFFrame *host;
    EFFrame *target;
    int idxInAll;

    RawResidualJacobian *J;

    VecNRf res_toZeroF;

#if defined(STEREO_MODE)
    Vec10f JpJdF;
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    Vec8f JpJdF;
#endif

    // status.
    bool isLinearized;

    // if residual is not OOB & not OUTLIER & should be used during accumulations
    bool isActiveAndIsGoodNEW;

    inline const bool &isActive() const { return isActiveAndIsGoodNEW; }
  };

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  class EFIMUResidual {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    inline EFIMUResidual(IMUResidual *org, EFSpeedAndBias *from_sb_, EFSpeedAndBias *to_sb_,
                         EFFrame *from_f_, EFFrame *to_f_) :
        data(org), from_sb(from_sb_), to_sb(to_sb_), from_f(from_f_), to_f(to_f_) {
      isLinearized = false;
      flaggedForMarginalization = false;

      J = new RawIMUResidualJacobian();
    }

    inline ~EFIMUResidual() {
      delete J;
    }

    void fixLinearizationF(EnergyFunctional *ef);

    void takeDataF();

    int fromSBIDX, toSBIDX;
    EFSpeedAndBias* from_sb;
    EFSpeedAndBias* to_sb;
    EFFrame* from_f;
    EFFrame* to_f;
    IMUResidual *data;
    int idxInAll;

    RawIMUResidualJacobian* J;

    Vec15f res_toZeroF;

    // [0] xi0, s0, [1] xi0, s1, [2] xi1, s0, [3] xi1, s1
    Mat69f JxiJsF[4];

    bool isLinearized;

    bool flaggedForMarginalization;
  };
#endif

  enum EFPointStatus {
    PS_GOOD = 0, PS_MARGINALIZE, PS_DROP
  };

  class EFPoint {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EFPoint(PointHessian *d, EFFrame *host_) : data(d), host(host_) {
      takeData();
      stateFlag = EFPointStatus::PS_GOOD;
      Hdd_accAF = 0.f;
      Hdd_accLF = 0.f;
    }

    void takeData();

    PointHessian *data;


    float priorF;
    float deltaF;


    // constant info (never changes in-between).
    int idxInPoints;
    EFFrame *host;

    // contains all residuals.
    std::vector<EFResidual *> residualsAll;

    float bdSumF;
    float HdiF;
    float Hdd_accLF;
    VecCf Hcd_accLF;
    float bd_accLF;
    float Hdd_accAF;
    VecCf Hcd_accAF;
    float bd_accAF;


    EFPointStatus stateFlag;
  };

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  class EFSpeedAndBias {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EFSpeedAndBias(SpeedAndBiasHessian* d) : data(d) {
      takeData();
      stateFlag = GOOD;
      bsSumF.setZero();
      HsiF.setZero();
      Hss_accLF.setZero();
      bs_accLF.setZero();
      Hss_accAF.setZero();
      bs_accAF.setZero();
    }

    ~EFSpeedAndBias() {
      for (EFIMUResidual *r : residualsAll)
        delete r;
      residualsAll.clear();
    }

    void takeData();

    SpeedAndBias priorF;
    SpeedAndBias deltaF;

    Vec9f bsSumF;
    Mat99f HsiF; //- inverse of (Hss_accLF + Hss_accAF)
    Mat99f Hss_accLF;
    Vec9f bs_accLF;
    Mat99f Hss_accAF;
    Vec9f bs_accAF;

    SpeedAndBiasHessian *data;
    int idx;

    std::vector<EFIMUResidual *> residualsAll; //- to as host

    enum EFSpeedAndBiasStatus {
      GOOD = 0, MARGINALIZE, DROP
    };

    EFSpeedAndBiasStatus stateFlag;
  };
#endif

  class EFFrame {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EFFrame(FrameHessian *d) : data(d) {
      takeData();
    }

    void takeData();

#if defined(STEREO_MODE)
    Vec10 prior;        // prior hessian (diagonal)
    Vec10 delta_prior;    // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
    Vec10 delta;        // state - state_zero.
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    Vec8 prior;        // prior hessian (diagonal)
    Vec8 delta_prior;    // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
    Vec8 delta;        // state - state_zero.
#endif


    std::vector<EFPoint *> points;
    FrameHessian *data;
    int idx;  // idx in frames.

    int frameID;
  };

}

