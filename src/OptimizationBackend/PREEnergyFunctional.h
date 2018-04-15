//
// Created by jg on 18-4-12.
//
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
#include "util/IndexThreadReduce.h"
#include "vector"
#include <math.h>
#include "map"


namespace dso {
#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  class CalibHessian;

  class FrameHessian;

  class IMUResidual;

  class EFIMUResidual;

  class EFSpeedAndBias;

  class SpeedAndBiasHessian;

  class EFResidual;

  class EFFrame;

  class EnergyFunctional;

  class AccumulatedTopHessian;

  class AccumulatedTopHessianSSE;

  class AccumulatedSCHessian;

  class AccumulatedSCHessianSSE;

  class CoarseTracker;

  extern bool PRE_EFAdjointsValid;
  extern bool PRE_EFIndicesValid;
  extern bool PRE_EFDeltaValid;


  class PREEnergyFunctional {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    friend class EFFrame;

    friend class EFResidual;

    friend class AccumulatedTopHessian;

    friend class AccumulatedTopHessianSSE;

    friend class AccumulatedSCHessian;

    friend class AccumulatedSCHessianSSE;

    PREEnergyFunctional();

    ~PREEnergyFunctional();

    void clear();

    EFIMUResidual *insertIMUResidual(IMUResidual *r);

    EFSpeedAndBias *insertSpeedAndBias(SpeedAndBiasHessian *sh);

    EFFrame *insertFrame(FrameHessian *fh);

    void marginalizeFrame(EFFrame *efF);

    void marginalizeSpeedAndBiasesF();

    void marginalizePointsF(const Mat1010 &M_last, const Vec10 &Mb_last,
                            const Mat1010 &Msc_last, const Vec10 &Mbsc_last);

    void dropFrame(EFFrame *efF);

    void dropIMUResidual(EFIMUResidual *r);

    void solveSystemF(int iteration, double lambda, Mat1010 &H_last, Vec10 &b_last);

    double calcMEnergyF();

    double calcLEnergyF_MT();

    void makeIDX();

    void setDeltaF();

    void setAdjointsF();

    std::vector<EFFrame *> frames;
    int nFrames;

    std::vector<EFSpeedAndBias *> speedAndBiases;
    int nSpeedAndBiases, nIMUResiduals, nMargSpeedAndBiases;

    MatXX HM;
    VecX bM;

    int resInA, resInL, resInM;
    MatXX lastHS;
    VecX lastbS;
    VecX lastX;
    std::vector<VecX> lastNullspaces_forLogging;
    std::vector<VecX> lastNullspaces_pose;
    std::vector<VecX> lastNullspaces_scale;
    std::vector<VecX> lastNullspaces_affA;
    std::vector<VecX> lastNullspaces_affB;

    IndexThreadReduce<Vec10> *red;

  private:

    VecX getStitchedDeltaF() const;

    void resubstituteF_MT(VecX x, bool MT);

    void accumulateIMUAF_MT(MatXX &H, VecX &b, bool MT);

    void accumulateIMULF_MT(MatXX &H, VecX &b, bool MT);

    void accumulateIMUSCF_MT(MatXX &H, VecX &b, MatXX &Hss_inv, MatXX &Hsx, VecX &bsr, bool MT);

    void accumulateIMUMF_MT(MatXX &H, VecX &b, bool MT);

    void accumulateIMUMSCF_MT(MatXX &H, VecX &b, bool MT);

    void orthogonalize(VecX *b, MatXX *H);

    Mat110f *adHTdeltaF;

    Mat1010 *adHost;
    Mat1010 *adTarget;

    Mat1010f *adHostF;
    Mat1010f *adTargetF;

    VecC cPrior;
    VecCf cPriorF;

    float currentLambda;
  };

#endif
}

