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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {


  bool EFAdjointsValid = false;
  bool EFIndicesValid = false;
  bool EFDeltaValid = false;

#if defined(STEREO_MODE)

  void EnergyFunctional::setAdjointsF(CalibHessian *Hcalib) {

    if (adHost != 0) delete[] adHost;
    if (adTarget != 0) delete[] adTarget;
    adHost = new Mat1010[nFrames * nFrames];
    adTarget = new Mat1010[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        FrameHessian *host = frames[h]->data;
        FrameHessian *target = frames[t]->data;

        SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

        Mat1010 AH = Mat1010::Identity();
        Mat1010 AT = Mat1010::Identity();

        AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
        AT.topLeftCorner<6, 6>() = Mat66::Identity();


        Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
                                                  target->aff_g2l_0()).cast<float>();

        if (t == h) {
          affLL = AffLight::fromToVecExposure(host->ab_exposure, host->rightFrame->ab_exposure, host->aff_g2l_0(),
                                              host->aff_g2l_r_0()).cast<float>();
        }
        //- original
//        AT(6, 6) = -affLL[0];
//        AH(6, 6) = affLL[0];
//        AT(7, 7) = -1;
//        AH(7, 7) = affLL[0];

        //- same as engel
//        if (t == h) {
//          AH(6, 8) = affLL[0];
//          AH(7, 9) = affLL[0];
//          AH(6, 6) = 0;
//          AH(7, 7) = 0;
//          AH(8, 8) = 0;
//          AH(9, 9) = 0;
//
//          AT(6, 6) = 0;
//          AT(7, 7) = 0;
//          AT(8, 8) = -affLL[0];
//          AT(9, 9) = -1;
//        }
//        else {
//          AT(6, 6) = -affLL[0];
//          AH(6, 6) = affLL[0];
//          AT(7, 7) = -1;
//          AH(7, 7) = affLL[0];
//
//          AT(8, 8) = 0;
//          AH(8, 8) = 0;
//          AT(9, 9) = 0;
//          AH(9, 9) = 0;
//        }

        //- opposite to engel
        if (t == h) {
          AH(6, 8) = -affLL[0];
          AH(7, 9) = -affLL[0];
          AH(6, 6) = 0;
          AH(7, 7) = 0;
          AH(8, 8) = 0;
          AH(9, 9) = 0;

          AT(6, 6) = 0;
          AT(7, 7) = 0;
          AT(8, 8) = affLL[0];
          AT(9, 9) = 1;
        }
        else {
          AT(6, 6) = affLL[0];
          AH(6, 6) = -affLL[0];
          AT(7, 7) = 1;
          AH(7, 7) = -affLL[0];

          AT(8, 8) = 0;
          AH(8, 8) = 0;
          AT(9, 9) = 0;
          AH(9, 9) = 0;
        }

        AH.block<3, 10>(0, 0) *= SCALE_XI_TRANS;
        AH.block<3, 10>(3, 0) *= SCALE_XI_ROT;
        AH.block<1, 10>(6, 0) *= SCALE_A;
        AH.block<1, 10>(7, 0) *= SCALE_B;
        AH.block<1, 10>(8, 0) *= SCALE_A;
        AH.block<1, 10>(9, 0) *= SCALE_B;
        AT.block<3, 10>(0, 0) *= SCALE_XI_TRANS;
        AT.block<3, 10>(3, 0) *= SCALE_XI_ROT;
        AT.block<1, 10>(6, 0) *= SCALE_A;
        AT.block<1, 10>(7, 0) *= SCALE_B;
        AT.block<1, 10>(8, 0) *= SCALE_A;
        AT.block<1, 10>(9, 0) *= SCALE_B;

        adHost[h + t * nFrames] = AH;
        adTarget[h + t * nFrames] = AT;
      }
    cPrior = VecC::Constant(setting_initialCalibHessian);


    if (adHostF != 0) delete[] adHostF;
    if (adTargetF != 0) delete[] adTargetF;
    adHostF = new Mat1010f[nFrames * nFrames];
    adTargetF = new Mat1010f[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
        adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
      }

    cPriorF = cPrior.cast<float>();


    EFAdjointsValid = true;
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EnergyFunctional::setAdjointsF(CalibHessian *Hcalib) {

    if (adHost != 0) delete[] adHost;
    if (adTarget != 0) delete[] adTarget;
    adHost = new Mat88[nFrames * nFrames];
    adTarget = new Mat88[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        FrameHessian *host = frames[h]->data;
        FrameHessian *target = frames[t]->data;

        SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

        Mat88 AH = Mat88::Identity();
        Mat88 AT = Mat88::Identity();

        AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
        AT.topLeftCorner<6, 6>() = Mat66::Identity();


        Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
                                                  target->aff_g2l_0()).cast<float>();
        //- original
//        AT(6, 6) = -affLL[0];
//        AH(6, 6) = affLL[0];
//        AT(7, 7) = -1;
//        AH(7, 7) = affLL[0];

        AT(6, 6) = affLL[0];
        AH(6, 6) = -affLL[0];
        AT(7, 7) = 1;
        AH(7, 7) = -affLL[0];

        AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
        AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
        AH.block<1, 8>(6, 0) *= SCALE_A;
        AH.block<1, 8>(7, 0) *= SCALE_B;
        AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
        AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
        AT.block<1, 8>(6, 0) *= SCALE_A;
        AT.block<1, 8>(7, 0) *= SCALE_B;

        adHost[h + t * nFrames] = AH;
        adTarget[h + t * nFrames] = AT;
      }
    cPrior = VecC::Constant(setting_initialCalibHessian);


    if (adHostF != 0) delete[] adHostF;
    if (adTargetF != 0) delete[] adTargetF;
    adHostF = new Mat88f[nFrames * nFrames];
    adTargetF = new Mat88f[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
        adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
      }

    cPriorF = cPrior.cast<float>();


    EFAdjointsValid = true;
  }

#endif

  EnergyFunctional::EnergyFunctional() {
    adHost = 0;
    adTarget = 0;
    marginalizeCountforDebug = 0;

    red = 0;

    adHostF = 0;
    adTargetF = 0;
    adHTdeltaF = 0;

    nFrames = nResiduals = nPoints = 0;
#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    nSpeedAndBiases = nIMUResiduals = nMargSpeedAndBiases = 0;
#endif

    HM = MatXX::Zero(CPARS, CPARS);
    bM = VecX::Zero(CPARS);


    accSSE_top_L = new AccumulatedTopHessianSSE();
    accSSE_top_A = new AccumulatedTopHessianSSE();
    accSSE_bot = new AccumulatedSCHessianSSE();

    resInA = resInL = resInM = 0;
    currentLambda = 0;
  }

  EnergyFunctional::~EnergyFunctional() {
    for (EFFrame *f : frames) {
      for (EFPoint *p : f->points) {
        for (EFResidual *r : p->residualsAll) {
          r->data->efResidual = 0;
          delete r;
        }
        p->data->efPoint = 0;
        delete p;
      }
      f->data->efFrame = 0;
      delete f;
    }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    for (EFSpeedAndBias *s : speedAndBiases) {
      s->data->efSB = 0;
      delete s;
    }
#endif

    if (adHost != 0) delete[] adHost;
    if (adTarget != 0) delete[] adTarget;


    if (adHostF != 0) delete[] adHostF;
    if (adTargetF != 0) delete[] adTargetF;
    if (adHTdeltaF != 0) delete[] adHTdeltaF;


    delete accSSE_top_L;
    delete accSSE_top_A;
    delete accSSE_bot;
  }

#if defined(STEREO_MODE)

  void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
    if (adHTdeltaF != 0) delete[] adHTdeltaF;
    adHTdeltaF = new Mat110f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        int idx = h + t * nFrames;
        adHTdeltaF[idx] =
            frames[h]->data->get_state_minus_stateZero().head<10>().cast<float>().transpose() * adHostF[idx]
            + frames[t]->data->get_state_minus_stateZero().head<10>().cast<float>().transpose() * adTargetF[idx];
      }

    cDeltaF = HCalib->value_minus_value_zero.cast<float>();
    for (EFFrame *f : frames) {
      f->delta = f->data->get_state_minus_stateZero().head<10>();
      f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<10>();

      for (EFPoint *p : f->points)
        p->deltaF = p->data->idepth - p->data->idepth_zero;
    }

    EFDeltaValid = true;
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EnergyFunctional::setDeltaF(CalibHessian *HCalib) {
    if (adHTdeltaF != 0) delete[] adHTdeltaF;
    adHTdeltaF = new Mat18f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        int idx = h + t * nFrames;
        adHTdeltaF[idx] =
            frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
            + frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
      }

    cDeltaF = HCalib->value_minus_value_zero.cast<float>();
    for (EFFrame *f : frames) {
      f->delta = f->data->get_state_minus_stateZero().head<8>();
      f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

      for (EFPoint *p : f->points)
        p->deltaF = p->data->idepth - p->data->idepth_zero;
    }

    EFDeltaValid = true;
  }

#endif

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  void EnergyFunctional::accumulateIMUAF_MT(MatXX &H, VecX &b, bool MT) {
    // Use one thread first.
    H = MatXX::Zero(nFrames * 10 + CPARS, nFrames * 10 + CPARS);
    b = VecX::Zero(nFrames * 10 + CPARS);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (r->isLinearized) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = J->resF;

        int fIdx = CPARS + 10 * r->from_f->idx;
        int tIdx = CPARS + 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, fIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(fIdx, fIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();
        b.segment<6>(fIdx).noalias() = (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() = (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void EnergyFunctional::accumulateIMULF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10 + CPARS, nFrames * 10 + CPARS);
    b = VecX::Zero(nFrames * 10 + CPARS);

    for(EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (!r->isLinearized) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = r->res_toZeroF;

        int fIdx = CPARS + 10 * r->from_f->idx;
        int tIdx = CPARS + 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, fIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(fIdx, fIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();
        b.segment<6>(fIdx).noalias() = (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() = (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void EnergyFunctional::accumulateIMUSCF_MT(MatXX &H, VecX &b, MatXX &Hss_inv, MatXX &Hsx, VecX &bsr, bool MT) {
    // Take IMU residuals' derivatives with respect to SpeedAndBiases,
    // the matrix is not diagonal for the reason that one residual associates two speedAndBiases
    MatXXf Hssf = MatXXf::Zero(nFrames * 9, nFrames * 9);
    MatXXf Hxsf = MatXXf::Zero(nFrames * 6, nFrames * 9);
    VecXf bsrf = VecXf::Zero(nFrames * 9);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        RawIMUResidualJacobian *J = r->J;

        int fIdxRaw = r->from_f->idx;
        int tIdxRaw = r->to_f->idx;
        Hssf.block<9, 9>(fIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[0];
        Hssf.block<9, 9>(tIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[1];
        Hssf.block<9, 9>(fIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[1];
        Hssf.block<9, 9>(tIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[0];

        Hxsf.block<6, 9>(fIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[0];
        Hxsf.block<6, 9>(tIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[1];
        Hxsf.block<6, 9>(fIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[1];
        Hxsf.block<6, 9>(tIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[0];
        Eigen::Matrix<float, 15, 1> resApprox;

        if (r->isLinearized)
          resApprox = r->res_toZeroF;
        else
          resApprox = J->resF;

        bsrf.segment<9>(fIdxRaw * 9).noalias() = J->Jrdsb[0].transpose() * resApprox;
        bsrf.segment<9>(tIdxRaw * 9).noalias() = J->Jrdsb[1].transpose() * resApprox;
      }
    }
    H = MatXX::Zero(nFrames * 10 + CPARS, nFrames * 10 + CPARS);
    b = VecX::Zero(nFrames * 10 + CPARS);

    MatXXf Hssf_inv = Hssf.inverse();//- This inverse is unavoidable for me now, unless fix one speedAndBias point.

    MatXXf H_small = Hxsf * Hssf_inv * Hxsf.transpose(); //- nFrame * 6
    VecXf b_small = Hxsf * Hssf_inv * bsrf; //- nFrame * 6

    //- save result
    for (int r = 0; r < nFrames; r++) {
      int rIdx = 10 * r + CPARS;
      H.block<6, 6>(rIdx, rIdx).noalias() = H_small.block<6, 6>(r * 6, r * 6).cast<double>();
      b.segment<6>(rIdx).noalias() = b_small.segment<6>(r * 6).cast<double>();
      for (int c = r + 1; c < nFrames; c++) {
        int cIdx = 10 * c + CPARS;

        H.block<6, 6>(rIdx, cIdx).noalias() = H_small.block<6, 6>(r * 6, c * 6).cast<double>();
        H.block<6, 6>(cIdx, rIdx).noalias() = H_small.block<6, 6>(c * 6, r * 6).cast<double>();
      }
    }
    Hss_inv = Hssf_inv.cast<double>();
    Hsx = Hxsf.transpose().cast<double>();
    bsr = bsrf.cast<double>();
  }

  void EnergyFunctional::accumulateIMUMF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10 + CPARS, nFrames * 10 + CPARS);
    b = VecX::Zero(nFrames * 10 + CPARS);

    for(EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (!r->flaggedForMarginalization) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = r->res_toZeroF;

        int fIdx = CPARS + 10 * r->from_f->idx;
        int tIdx = CPARS + 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, fIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(fIdx, fIdx).noalias() = (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() = (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();
        b.segment<6>(fIdx).noalias() = (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() = (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void EnergyFunctional::accumulateIMUMSCF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10 + CPARS, nFrames * 10 + CPARS);
    b = VecX::Zero(nFrames * 10 + CPARS);

    if (nMargSpeedAndBiases == 0) return;

    int *pair = new int[nMargSpeedAndBiases];
    MatXXf Hssf = MatXXf::Zero(nMargSpeedAndBiases * 9, nMargSpeedAndBiases * 9);
    MatXXf Hxsf = MatXXf::Zero(nMargSpeedAndBiases * 6, nMargSpeedAndBiases * 9);
    VecXf bsrf = VecXf::Zero(nMargSpeedAndBiases * 9);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (!r->flaggedForMarginalization) continue;
        RawIMUResidualJacobian *J = r->J;

        int fIdxRaw = r->from_sb->data->margIDX;
        int tIdxRaw = r->to_sb->data->margIDX;
        pair[fIdxRaw] = r->from_sb->idx;
        pair[tIdxRaw] = r->to_sb->idx;
        Hssf.block<9, 9>(fIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[0];
        Hssf.block<9, 9>(tIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[1];
        Hssf.block<9, 9>(fIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[1];
        Hssf.block<9, 9>(tIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[0];

        Hxsf.block<6, 9>(fIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[0];
        Hxsf.block<6, 9>(tIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[1];
        Hxsf.block<6, 9>(fIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[1];
        Hxsf.block<6, 9>(tIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[0];
        Eigen::Matrix<float, 15, 1> resApprox;

        if (r->isLinearized)
          resApprox = r->res_toZeroF;
        else
          resApprox = J->resF;

        bsrf.segment<9>(fIdxRaw * 9).noalias() = J->Jrdsb[0].transpose() * resApprox;
        bsrf.segment<9>(tIdxRaw * 9).noalias() = J->Jrdsb[1].transpose() * resApprox;
      }
    }

    MatXXf Hssf_inv = Hssf.inverse();//- This inverse is unavoidable for me now, unless fix one speedAndBias point.

    LOG(INFO) << "Hssf: " << Hssf;

    MatXXf H_small = Hxsf * Hssf_inv * Hxsf.transpose(); //- nFrame * 6
    VecXf b_small = Hxsf * Hssf_inv * bsrf; //- nFrame * 6

    //- save result
    for (int r = 0; r < nMargSpeedAndBiases; r++) {
      int rBIdx = 10 * pair[r] + CPARS;
      H.block<6, 6>(rBIdx, rBIdx).noalias() = H_small.block<6, 6>(r * 6, r * 6).cast<double>();
      b.segment<6>(rBIdx).noalias() = b_small.segment<6>(r * 6).cast<double>();
      for (int c = r + 1; c < nMargSpeedAndBiases; c++) {
        int cBIdx = 10 * pair[c] + CPARS;

        H.block<6, 6>(rBIdx, cBIdx).noalias() = H_small.block<6, 6>(r * 6, c * 6).cast<double>();
        H.block<6, 6>(cBIdx, rBIdx).noalias() = H_small.block<6, 6>(c * 6, r * 6).cast<double>();
      }
    }

    delete [] pair;
  }

#endif

// accumulates & shifts L.
  void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
      red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0, 0, 0);
      red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                              accSSE_top_A, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
      accSSE_top_A->stitchDoubleMT(red, H, b, this, false, true);
      resInA = accSSE_top_A->nres[0];
    }
    else {
      accSSE_top_A->setZero(nFrames);
      for (EFFrame *f : frames)
        for (EFPoint *p : f->points)
          accSSE_top_A->addPoint<0>(p, this); // 0 = active, 1 = linearized, 2=marginalize
      accSSE_top_A->stitchDoubleMT(red, H, b, this, false,
                                   false); // IndexThreadReduce<Vec10>* red, MatXX &H, VecX &b, EnergyFunctional const * const EF, bool usePrior, bool MT
      resInA = accSSE_top_A->nres[0];
    }
  }

// accumulates & shifts L.
  void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
      red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0, 0, 0);
      red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                              accSSE_top_L, &allPoints, this, _1, _2, _3, _4), 0, allPoints.size(), 50);
      accSSE_top_L->stitchDoubleMT(red, H, b, this, true, true);
      resInL = accSSE_top_L->nres[0];
    }
    else {
      accSSE_top_L->setZero(nFrames);
      for (EFFrame *f : frames)
        for (EFPoint *p : f->points)
          accSSE_top_L->addPoint<1>(p, this); // 0 = active, 1 = linearized, 2=marginalize
      accSSE_top_L->stitchDoubleMT(red, H, b, this, true, false);
      resInL = accSSE_top_L->nres[0];
    }
  }


  void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
    if (MT) {
      red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0, 0);
      red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                              accSSE_bot, &allPoints, true, _1, _2, _3, _4), 0, allPoints.size(), 50);
      accSSE_bot->stitchDoubleMT(red, H, b, this, true);
    }
    else {
      accSSE_bot->setZero(nFrames);
      for (EFFrame *f : frames)
        for (EFPoint *p : f->points)
          accSSE_bot->addPoint(p, true);
      accSSE_bot->stitchDoubleMT(red, H, b, this, false);
    }
  }

  void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT) {
#if defined(STEREO_MODE)
    assert(x.size() == CPARS + nFrames * 10);
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    assert(x.size() == CPARS + nFrames * 8);
#endif
    VecXf xF = x.cast<float>();
    HCalib->step = x.head<CPARS>();

    VecCf cstep = xF.head<CPARS>();
    if (!std::isfinite(cstep.norm())) cstep.setZero(); // TODO: remove this to checkout if the system will be destoryed.
#if defined(STEREO_MODE)
    Mat110f *xAd = new Mat110f[nFrames * nFrames];
    for (EFFrame *h : frames) {
      h->data->step = x.segment<10>(CPARS + 10 * h->idx);

      for (EFFrame *t : frames)
        xAd[nFrames * h->idx + t->idx] =
            xF.segment<10>(CPARS + 10 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
            + xF.segment<10>(CPARS + 10 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    Mat18f *xAd = new Mat18f[nFrames * nFrames];
    for (EFFrame *h : frames) {
      h->data->step.head<8>() = x.segment<8>(CPARS + 8 * h->idx);
      h->data->step.tail<2>().setZero();// TODO: remove this to checkout if the system will be destoryed.

      for (EFFrame *t : frames)
        xAd[nFrames * h->idx + t->idx] =
            xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
            + xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }
#endif
    if (MT)
      red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
                              this, cstep, xAd, _1, _2, _3, _4), 0, allPoints.size(), 50);
    else
      resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

    delete[] xAd;
  }

#if defined(STEREO_MODE)

  void EnergyFunctional::resubstituteFPt(
      const VecCf &xc, Mat110f *xAd, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++) {
      EFPoint *p = allPoints[k];

      int ngoodres = 0;
      for (EFResidual *r : p->residualsAll) if (r->isActive()) ngoodres++;
      if (ngoodres == 0) {
        p->data->step = 0;
        continue;
      }

      float b = p->bdSumF;
      //- original I think this code snippet is wrong.
//      b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);
//
//      for (EFResidual *r : p->residualsAll) {
//        if (!r->isActive()) continue;
//        if (r->targetIDX == -1) { //- static stereo residual
//          b -= xAd[r->hostIDX * nFrames + r->hostIDX] * r->JpJdF;
//        }
//        else {
//          b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
//        }
//      }

      b += xc.dot(p->Hcd_accAF + p->Hcd_accLF);

      for (EFResidual *r : p->residualsAll) {
        if (!r->isActive()) continue;
        if (r->targetIDX == -1) { //- static stereo residual
          b += xAd[r->hostIDX * nFrames + r->hostIDX] * r->JpJdF;
        }
        else {
          b += xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }
      }

      p->data->step = -b * p->HdiF;
      assert(std::isfinite(p->data->step));
    }
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EnergyFunctional::resubstituteFPt(
      const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++) {
      EFPoint *p = allPoints[k];

      int ngoodres = 0;
      for (EFResidual *r : p->residualsAll) if (r->isActive()) ngoodres++;
      if (ngoodres == 0) {
        p->data->step = 0;
        continue;
      }
      float b = p->bdSumF;
      //- original I think this code snippet is wrong.
//      b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);
//
//      for (EFResidual *r : p->residualsAll) {
//        if (!r->isActive()) continue;
//        if (r->targetIDX == -1) { //- static stereo residual
//          b -= xAd[r->hostIDX * nFrames + r->hostIDX] * r->JpJdF;
//        }
//        else {
//          b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
//        }
//      }
      b += xc.dot(p->Hcd_accAF + p->Hcd_accLF);

      for (EFResidual *r : p->residualsAll) {
        if (!r->isActive()) continue;
        if (r->targetIDX == -1) { //- static stereo residual
          b += xAd[r->hostIDX * nFrames + r->hostIDX] * r->JpJdF;
        }
        else {
          b += xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }
      }

      p->data->step = -b * p->HdiF;
//      assert(std::isfinite(p->data->step));
    }
  }

#endif

  double EnergyFunctional::calcMEnergyF() {

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    VecX delta = getStitchedDeltaF();
    return delta.dot(2 * bM + HM * delta);
  }

#if defined(STEREO_MODE)

  void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

    Accumulator1 E;
    E.initialize();
    VecCf dc = cDeltaF;

    for (int i = min; i < max; i++) {
      EFPoint *p = allPoints[i];
      float dd = p->deltaF;

      for (EFResidual *r : p->residualsAll) {
        if (!r->isLinearized || !r->isActive()) continue;

        Mat110f dp;
        if (r->targetIDX == -1) { //- static stereo residual
          dp = adHTdeltaF[r->hostIDX + nFrames * r->hostIDX];
        }
        else { //- temporal stereo residual
          dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
        }
        RawResidualJacobian *rJ = r->J;



        // compute Jp*delta
        float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
                             + rJ->Jpdc[0].dot(dc)
                             + rJ->Jpdd[0] * dd;

        float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
                             + rJ->Jpdc[1].dot(dc)
                             + rJ->Jpdd[1] * dd;

        __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
        __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
        __m128 delta_a = _mm_set1_ps((float) (dp[6]));
        __m128 delta_b = _mm_set1_ps((float) (dp[7]));
        __m128 delta_a_r = _mm_set1_ps((float) (dp[8]));
        __m128 delta_b_r = _mm_set1_ps((float) (dp[9]));

        for (int i = 0; i + 3 < patternNum; i += 4) {
          // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
          __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x);
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 2)) + i), delta_a_r));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 3)) + i), delta_b_r));

          __m128 r0 = _mm_load_ps(((float *) &r->res_toZeroF) + i);
          r0 = _mm_add_ps(r0, r0);
          r0 = _mm_add_ps(r0, Jdelta);
          Jdelta = _mm_mul_ps(Jdelta, r0);
          E.updateSSENoShift(Jdelta);
        }
        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
          float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
                         rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7] +
                         rJ->JabF[1][i] * dp[8] + rJ->JabF[3][i] * dp[9];
          E.updateSingleNoShift((float) (Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
        }
      }
      E.updateSingle(p->deltaF * p->deltaF * p->priorF);
    }
    E.finish();
    (*stats)[0] += E.A;
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10 *stats, int tid) {

    Accumulator1 E;
    E.initialize();
    VecCf dc = cDeltaF;

    for (int i = min; i < max; i++) {
      EFPoint *p = allPoints[i];
      float dd = p->deltaF;

      for (EFResidual *r : p->residualsAll) {
        if (!r->isLinearized || !r->isActive()) continue;

        Mat18f dp;
        if (r->targetIDX == -1) { //- static stereo residual
          dp = adHTdeltaF[r->hostIDX + nFrames * r->hostIDX];
        }
        else { //- temporal stereo residual
          dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
        }
        RawResidualJacobian *rJ = r->J;



        // compute Jp*delta
        float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>())
                             + rJ->Jpdc[0].dot(dc)
                             + rJ->Jpdd[0] * dd;

        float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>())
                             + rJ->Jpdc[1].dot(dc)
                             + rJ->Jpdd[1] * dd;

        __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
        __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
        __m128 delta_a = _mm_set1_ps((float) (dp[6]));
        __m128 delta_b = _mm_set1_ps((float) (dp[7]));

        for (int i = 0; i + 3 < patternNum; i += 4) {
          // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
          __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x);
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
          Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));

          __m128 r0 = _mm_load_ps(((float *) &r->res_toZeroF) + i);
          r0 = _mm_add_ps(r0, r0);
          r0 = _mm_add_ps(r0, Jdelta);
          Jdelta = _mm_mul_ps(Jdelta, r0);
          E.updateSSENoShift(Jdelta);
        }
        for (int i = ((patternNum >> 2) << 2); i < patternNum; i++) {
          float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
                         rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
          E.updateSingleNoShift((float) (Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
        }
      }
      E.updateSingle(p->deltaF * p->deltaF * p->priorF);
    }
    E.finish();
    (*stats)[0] += E.A;
  }

#endif

  double EnergyFunctional::calcLEnergyF_MT() {
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    double E = 0;
    for (EFFrame *f : frames)
      E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

    E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

    red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
                            this, _1, _2, _3, _4), 0, allPoints.size(), 50);

    return E + red->stats[0];
  }

  EFResidual *EnergyFunctional::insertResidual(PointFrameResidual *r) {
    EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
    efr->idxInAll = r->point->efPoint->residualsAll.size();
    r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t) efr->host->frameID) << 32) + ((uint64_t) efr->target->frameID)][0]++;

    nResiduals++;
    r->efResidual = efr;
    return efr;
  }


  EFResidual *EnergyFunctional::insertStaticResidual(PointFrameResidual *r) {
    assert(r->host->rightFrame == r->target);
    EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, 0);
    efr->idxInAll = r->point->efPoint->residualsAll.size();
    r->point->efPoint->residualsAll.push_back(efr);

    nResiduals++;
    r->efResidual = efr;
    return efr;
  }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  EFIMUResidual* EnergyFunctional::insertIMUResidual(IMUResidual *r) {
    EFIMUResidual *efr = new EFIMUResidual(r, r->from_sb->efSB, r->to_sb->efSB, r->from_sb->host->efFrame, r->to_sb->host->efFrame);
    efr->idxInAll = r->to_sb->efSB->residualsAll.size();
    r->to_sb->efSB->residualsAll.push_back(efr); //- toSpeedAndBias as host.

    nIMUResiduals++;
    r->efIMUResidual = efr;
    return efr;
  }

  EFSpeedAndBias* EnergyFunctional::insertSpeedAndBias(SpeedAndBiasHessian *sh) {
    EFSpeedAndBias* efs = new EFSpeedAndBias(sh);
    efs->idx = speedAndBiases.size();
    speedAndBiases.push_back(efs);

    nSpeedAndBiases++;
    sh->efSB = efs;
    return efs;
  }
#endif

  EFFrame *EnergyFunctional::insertFrame(FrameHessian *fh, CalibHessian *Hcalib) {
    EFFrame *eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff);

    nFrames++;
    fh->efFrame = eff;

#if defined(STEREO_MODE)
    assert(HM.cols() == 10 * nFrames + CPARS - 10);
    bM.conservativeResize(10 * nFrames + CPARS);
    HM.conservativeResize(10 * nFrames + CPARS, 10 * nFrames + CPARS);
    bM.tail<10>().setZero();
    HM.rightCols<10>().setZero();
    HM.bottomRows<10>().setZero();
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    assert(HM.cols() == 8 * nFrames + CPARS - 8);
    bM.conservativeResize(8 * nFrames + CPARS);
    HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    bM.tail<8>().setZero();
    HM.rightCols<8>().setZero();
    HM.bottomRows<8>().setZero();
#endif
    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    setAdjointsF(Hcalib);
    makeIDX();


    for (EFFrame *fh2 : frames) {
      connectivityMap[(((uint64_t) eff->frameID) << 32) + ((uint64_t) fh2->frameID)] = Eigen::Vector2i(0, 0);
      if (fh2 != eff)
        connectivityMap[(((uint64_t) fh2->frameID) << 32) + ((uint64_t) eff->frameID)] = Eigen::Vector2i(0, 0);
    }

    return eff;
  }

  EFPoint *EnergyFunctional::insertPoint(PointHessian *ph) {
    EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
    efp->idxInPoints = ph->host->efFrame->points.size();
    ph->host->efFrame->points.push_back(efp);

    nPoints++;
    ph->efPoint = efp;

    EFIndicesValid = false;

    return efp;
  }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  void EnergyFunctional::dropIMUResidual(EFIMUResidual *r) {
    EFSpeedAndBias *s = r->to_sb;
    assert(r == s->residualsAll[r->idxInAll]);

    s->residualsAll[r->idxInAll] = s->residualsAll.back();
    s->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    s->residualsAll.pop_back();

    nIMUResiduals--;
    r->data->efIMUResidual = 0;
    delete r;
  }
#endif

  void EnergyFunctional::dropResidual(EFResidual *r) {
    EFPoint *p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();


    if (r->isActive())
      r->host->data->shell->statistics_goodResOnThis++;
    else
      r->host->data->shell->statistics_outlierResOnThis++;

    if (r->targetIDX == -1) { //- static stereo residual
      //- do nothing
    }
    else { //- temporal stereo residual
      connectivityMap[(((uint64_t) r->host->frameID) << 32) + ((uint64_t) r->target->frameID)][0]--;
    }
    nResiduals--;
    r->data->efResidual = 0;
    delete r;
  }

#if defined(STEREO_MODE)

  void EnergyFunctional::marginalizeFrame(EFFrame *efF) {

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    LOG(INFO) << "marginalize times: " << marginalizeCountforDebug;

    assert((int) efF->points.size() == 0);
    int ndim = nFrames * 10 + CPARS - 10;// new dimension
    int odim = nFrames * 10 + CPARS;// old dimension

    MatXX HM_temp = HM;
    VecX bM_temp = bM;

    if ((int) efF->idx != (int) frames.size() - 1) {
      int io = efF->idx * 10 + CPARS;  // index of frame to move to end
      int ntail = 10 * (nFrames - efF->idx - 1);
      assert((io + 10 + ntail) == nFrames * 10 + CPARS);

      Vec10 bTmp = bM.segment<10>(io);
      VecX tailTMP = bM.tail(ntail);
      bM.segment(io, ntail) = tailTMP;
      bM.tail<10>() = bTmp;

      MatXX HtmpCol = HM.block(0, io, odim, 10);
      MatXX rightColsTmp = HM.rightCols(ntail);
      HM.block(0, io, odim, ntail) = rightColsTmp;
      HM.rightCols(10) = HtmpCol;

      MatXX HtmpRow = HM.block(io, 0, 10, odim);
      MatXX botRowsTmp = HM.bottomRows(ntail);
      HM.block(io, 0, ntail, odim) = botRowsTmp;
      HM.bottomRows(10) = HtmpRow;
    }

    // marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<10, 10>().diagonal() += efF->prior;
    bM.tail<10>() += efF->prior.cwiseProduct(efF->delta_prior);

    VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    // scale!
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled = SVecI.asDiagonal() * bM;

    // invert bottom part!
    Mat1010 hpi = HMScaled.bottomRightCorner<10, 10>();
    hpi = 0.5f * (hpi + hpi);
    hpi = hpi.inverse();
    hpi = 0.5f * (hpi + hpi);
    if (!std::isfinite(hpi.norm())) {
      LOG(INFO) << "hpi.norm() infinite";
      LOG(INFO) << "efF l_l : " << efF->data->aff_g2l().a << efF->data->aff_g2l().b;
      LOG(INFO) << "efF l_r : " << efF->data->aff_g2l_r().a << efF->data->aff_g2l_r().b;
      hpi = Mat1010::Zero();
      Mat88 botRht88;
      botRht88 = HMScaled.bottomRightCorner<10, 10>().topLeftCorner<8, 8>();
      if (botRht88.norm() != 0) {
        botRht88 = 0.5f * (botRht88 + botRht88);
        botRht88 = botRht88.inverse();
        botRht88 = 0.5f * (botRht88 + botRht88);
        hpi.block<8, 8>(0, 0) = botRht88;
      }
      LOG(INFO) << HMScaled.bottomRightCorner<10, 10>();
    }

    // schur-complement!
    MatXX bli = HMScaled.bottomLeftCorner(10, ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(10, ndim);
    bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<10>();

    //unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
    bM = bMScaled.head(ndim);

    // remove from vector, without changing the order!
    for (unsigned int i = efF->idx; i + 1 < frames.size(); i++) {
      frames[i] = frames[i + 1];
      frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    efF->data->efFrame = 0;

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    EFSpeedAndBias *efSB = efF->data->speedAndBiasHessian->efSB;
    for (unsigned int i = efSB->idx; i + 1 < speedAndBiases.size(); i++) {
      speedAndBiases[i] = speedAndBiases[i+1];
      speedAndBiases[i]->idx = i;
    }
    for (unsigned int i = 0; i < efSB->residualsAll.size(); i++)
      dropIMUResidual(efSB->residualsAll[i]);
    speedAndBiases.pop_back();
    nSpeedAndBiases--;
    efSB->data->efSB = 0;
    delete efSB;
#endif

    assert((int) frames.size() * 10 + CPARS == (int) HM.rows());
    assert((int) frames.size() * 10 + CPARS == (int) HM.cols());
    assert((int) frames.size() * 10 + CPARS == (int) bM.size());
    assert((int) frames.size() == (int) nFrames);

//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    makeIDX();
    delete efF;
    marginalizeCountforDebug++;
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EnergyFunctional::marginalizeFrame(EFFrame *efF) {

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    assert((int) efF->points.size() == 0);
    int ndim = nFrames * 8 + CPARS - 8;// new dimension
    int odim = nFrames * 8 + CPARS;// old dimension

//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());

    if ((int) efF->idx != (int) frames.size() - 1) {
      int io = efF->idx * 8 + CPARS;  // index of frame to move to end
      int ntail = 8 * (nFrames - efF->idx - 1);
      assert((io + 8 + ntail) == nFrames * 8 + CPARS);

      Vec8 bTmp = bM.segment<8>(io);
      VecX tailTMP = bM.tail(ntail);
      bM.segment(io, ntail) = tailTMP;
      bM.tail<8>() = bTmp;

      MatXX HtmpCol = HM.block(0, io, odim, 8);
      MatXX rightColsTmp = HM.rightCols(ntail);
      HM.block(0, io, odim, ntail) = rightColsTmp;
      HM.rightCols(8) = HtmpCol;

      MatXX HtmpRow = HM.block(io, 0, 8, odim);
      MatXX botRowsTmp = HM.bottomRows(ntail);
      HM.block(io, 0, ntail, odim) = botRowsTmp;
      HM.bottomRows(8) = HtmpRow;
    }

//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8, 8>().diagonal() += efF->prior;
    bM.tail<8>() += efF->prior.cwiseProduct(efF->delta_prior);

    VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    // scale!
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled = SVecI.asDiagonal() * bM;

    // invert bottom part!
    Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
    hpi = 0.5f * (hpi + hpi);
    hpi = hpi.inverse();
    hpi = 0.5f * (hpi + hpi);

    // schur-complement!
    MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
    bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

    //unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
    bM = bMScaled.head(ndim);

    // remove from vector, without changing the order!
    for (unsigned int i = efF->idx; i + 1 < frames.size(); i++) {
      frames[i] = frames[i + 1];
      frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    efF->data->efFrame = 0;

    assert((int) frames.size() * 8 + CPARS == (int) HM.rows());
    assert((int) frames.size() * 8 + CPARS == (int) HM.cols());
    assert((int) frames.size() * 8 + CPARS == (int) bM.size());
    assert((int) frames.size() == (int) nFrames);

//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    makeIDX();
    delete efF;
  }

#endif

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
  void EnergyFunctional::marginalizeSpeedAndBiasesF() {
    MatXX M_imu, Msc_imu;
    VecX Mb_imu, Mbsc_imu;

    accumulateIMUMF_MT(M_imu, Mb_imu, multiThreading);

    accumulateIMUMSCF_MT(Msc_imu, Mbsc_imu, multiThreading);

    MatXX H = M_imu - Msc_imu;
    VecX b = Mb_imu - Mbsc_imu;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
      bool haveFirstFrame = false;
      for (EFFrame *f : frames) if (f->frameID == 0) haveFirstFrame = true;

      if (!haveFirstFrame)
        orthogonalize(&b, &H);

    }

    HM += setting_margWeightFac * H;
    bM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
      orthogonalize(&bM, &HM);
  }
#endif

  void EnergyFunctional::marginalizePointsF() {
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);


    allPointsToMarg.clear();
    for (EFFrame *f : frames) {
      for (int i = 0; i < (int) f->points.size(); i++) {
        EFPoint *p = f->points[i];
        if (p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
          p->priorF *= setting_idepthFixPriorMargFac;
          for (EFResidual *r : p->residualsAll)
            if (r->isActive() && r->targetIDX != -1)
              connectivityMap[(((uint64_t) r->host->frameID) << 32) + ((uint64_t) r->target->frameID)][1]++;
          allPointsToMarg.push_back(p);
        }
      }
    }

    LOG(INFO) << "allPointsToMarg.size(): " << allPointsToMarg.size();
    //-- checkout if there is points with static residaul in allPointsToMarg
    int countIDX = 0;
    int countIDX1 = 0;
    for (EFPoint *p : allPointsToMarg) {
      for (EFResidual *r : p->residualsAll) {
        if (r->isActive()) {
          if (r->targetIDX == -1) {
            countIDX++;
            if (r->point->host->data->flaggedForMarginalization) countIDX1++;
          }
        }
      }
    }
    LOG(INFO) << "EnergyFunctional::marginalizePointsF countIDX " << countIDX1 << " / " << countIDX;

    accSSE_bot->setZero(nFrames);
    accSSE_top_A->setZero(nFrames);
    for (EFPoint *p : allPointsToMarg) {
      accSSE_top_A->addPoint<2>(p, this); // 0 = active, 1 = linearized, 2=marginalize
      accSSE_bot->addPoint(p, false);
      removePoint(p);
    }
    MatXX M, Msc;
    VecX Mb, Mbsc;
    accSSE_top_A->stitchDouble(M, Mb, this, false, false);
    accSSE_bot->stitchDouble(Msc, Mbsc, this);

    resInM += accSSE_top_A->nres[0];

    MatXX H = M - Msc;
    VecX b = Mb - Mbsc;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
      // have a look if prior is there.
      bool haveFirstFrame = false;
      for (EFFrame *f : frames) if (f->frameID == 0) haveFirstFrame = true;

      if (!haveFirstFrame)
        orthogonalize(&b, &H);

    }

//	printf("HM.size(): %ld, %ld\n", HM.cols(), HM.rows());

    HM += setting_margWeightFac * H;
    bM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
      orthogonalize(&bM, &HM);


//    LOG(INFO) << "HM: " << HM;
//    LOG(INFO) << "M: " << M;
//    LOG(INFO) << "Msc: " << Msc;
//    LOG(INFO) << HM.cols() << ", " << HM.rows();

    EFIndicesValid = false;
    makeIDX();
  }

  void EnergyFunctional::dropPointsF() {


    for (EFFrame *f : frames) {
      for (int i = 0; i < (int) f->points.size(); i++) {
        EFPoint *p = f->points[i];
        if (p->stateFlag == EFPointStatus::PS_DROP) {
          removePoint(p);
          i--;
        }
      }
    }

    EFIndicesValid = false;
    makeIDX();
  }


  void EnergyFunctional::removePoint(EFPoint *p) {
    for (EFResidual *r : p->residualsAll)
      dropResidual(r);

    EFFrame *h = p->host;
    h->points[p->idxInPoints] = h->points.back();
    h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
    h->points.pop_back();

    nPoints--;
    p->data->efPoint = 0;

    EFIndicesValid = false;

    delete p;
  }

  void EnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";


    // decide to which nullspaces to orthogonalize.
    std::vector<VecX> ns;
    ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
//    ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
    if (setting_affineOptModeA <= 0)
      ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
    if (setting_affineOptModeB <= 0)
      ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());





    // make Nullspaces matrix
    MatXX N(ns[0].rows(), ns.size());
    for (unsigned int i = 0; i < ns.size(); i++)
      N.col(i) = ns[i].normalized();



    // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
    Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX SNN = svdNN.singularValues();
    double minSv = 1e10, maxSv = 0;
    for (int i = 0; i < SNN.size(); i++) {
      if (SNN[i] < minSv) minSv = SNN[i];
      if (SNN[i] > maxSv) maxSv = SNN[i];
    }
    for (int i = 0; i < SNN.size(); i++) {
      if (SNN[i] > setting_solverModeDelta * maxSv)
        SNN[i] = 1.0 / SNN[i];
      else SNN[i] = 0;
    }

    MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose();  // [dim] x 9.
    MatXX NNpiT = N * Npi.transpose();  // [dim] x [dim].
    MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());  // = N * (N' * N)^-1 * N'.

    if (b != 0) *b -= NNpiTS * *b;
    if (H != 0) *H -= NNpiTS * *H * NNpiTS;


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

  }


  void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian *HCalib) {
    if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
    if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    MatXX HL_top, HA_top, H_sc;
    VecX bL_top, bA_top, bM_top, b_sc;

    // active
    accumulateAF_MT(HA_top, bA_top, multiThreading);

    // linearized
    accumulateLF_MT(HL_top, bL_top, multiThreading);

    // schur complement
    accumulateSCF_MT(H_sc, b_sc, multiThreading);

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    MatXX HL_top_imu, HA_top_imu, H_sc_imu;
    VecX bL_top_imu, bA_top_imu, b_sc_imu;
    //- For speedAndBiases steps compute
    MatXX Hss_inv, Hsx;
    VecX bsr;

    accumulateIMUAF_MT(HA_top_imu, bA_top_imu, multiThreading);

    accumulateIMULF_MT(HL_top_imu, bL_top_imu, multiThreading);

    accumulateIMUSCF_MT(H_sc_imu, b_sc_imu, Hss_inv, Hsx, bsr, multiThreading);

    HA_top += HA_top_imu;
    HL_top += HL_top_imu;
    H_sc += H_sc_imu;

    bA_top += bA_top_imu;
    bL_top += bL_top_imu;
    b_sc += b_sc_imu;
#endif

    bM_top = (bM + HM * getStitchedDeltaF());

    MatXX HFinal_top;
    VecX bFinal_top;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
      // have a look if prior is there.
      bool haveFirstFrame = false;
      for (EFFrame *f : frames) if (f->frameID == 0) haveFirstFrame = true;


      MatXX HT_act = HL_top + HA_top - H_sc;
      VecX bT_act = bL_top + bA_top - b_sc;


      if (!haveFirstFrame)
        orthogonalize(&bT_act, &HT_act);

      HFinal_top = HT_act + HM;
      bFinal_top = bT_act + bM_top;


      lastHS = HFinal_top;
      lastbS = bFinal_top;
#if defined(STEREO_MODE)
      for (int i = 0; i < 10 * nFrames + CPARS; i++) HFinal_top(i, i) *= (1 + lambda);
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
      for (int i = 0; i < 8 * nFrames + CPARS; i++) HFinal_top(i, i) *= (1 + lambda);
#endif
    }
    else { //- here


      HFinal_top = HL_top + HM + HA_top;
      bFinal_top = bL_top + bM_top + bA_top - b_sc;

      lastHS = HFinal_top - H_sc;
      lastbS = bFinal_top;

#if defined(STEREO_MODE)
      for (int i = 0; i < 10 * nFrames + CPARS; i++) HFinal_top(i, i) *= (1 + lambda);
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
      for (int i = 0; i < 8 * nFrames + CPARS; i++) HFinal_top(i, i) *= (1 + lambda);
#endif
      HFinal_top -= H_sc * (1.0f / (1 + lambda));
    }


    VecX x;
    if (setting_solverMode & SOLVER_SVD) {
      VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
      MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
      VecX bFinalScaled = SVecI.asDiagonal() * -bFinal_top;
      Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

      VecX S = svd.singularValues();
      double minSv = 1e10, maxSv = 0;
      for (int i = 0; i < S.size(); i++) {
        if (S[i] < minSv) minSv = S[i];
        if (S[i] > maxSv) maxSv = S[i];
      }

      VecX Ub = svd.matrixU().transpose() * bFinalScaled;
      int setZero = 0;
      for (int i = 0; i < Ub.size(); i++) {
        if (S[i] < setting_solverModeDelta * maxSv) {
          Ub[i] = 0;
          setZero++;
        }

        if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7)) {
          Ub[i] = 0;
          setZero++;
        }
        else Ub[i] /= S[i];
      }
      x = SVecI.asDiagonal() * svd.matrixV() * Ub;

    }
    else { //- here
      VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
      MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
      x = SVecI.asDiagonal() *
          HFinalScaled.ldlt().solve(SVecI.asDiagonal() * -bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
      if (!std::isfinite(x.norm())) {
//        LOG(INFO) << "HFinal_top: " << HFinal_top;
//        LOG(INFO) << "bFinal_top: " << bFinal_top;
//        LOG(INFO) << "H_sc: " << H_sc;
//        LOG(INFO) << "HM: " << HM;
//        LOG(INFO) << "HA_top_imu : " << HA_top_imu;
//        LOG(INFO) << "HL_top_imu : " << HL_top_imu;
//        LOG(INFO) << "bA_top_imu : " << bA_top_imu;
//        LOG(INFO) << "bL_top_imu : " << bL_top_imu;
//        LOG(INFO) << "H_sc_imu : " << H_sc_imu;
//        LOG(INFO) << "b_sc_imu : " << b_sc_imu;

//        bM_top = (bM + HM * getStitchedDeltaF());
//        HFinal_top = HL_top + HM + HA_top;
//        bFinal_top = bL_top + bM_top + bA_top - b_sc;
      }
    }

    if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
        (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
      VecX xOld = x;
      orthogonalize(&x, 0);
    }


    lastX = x;

    currentLambda = lambda;
    //- Modify step.
    resubstituteF_MT(x, HCalib, multiThreading);
    currentLambda = 0;
#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    //- resubstitute speedAndBiases
    //- Prepare \delta_X first
    VecX deltaX = VecX::Zero(nFrames * 6);
    for (int r = 0; r < nFrames; r++) {
      int rIdx = r * 10 + CPARS;
      deltaX.segment<6>(r * 6) = x.segment<6>(rIdx);
    }
    VecX deltaS = -Hss_inv * (bsr + Hsx * deltaX);
    for(EFSpeedAndBias *s : speedAndBiases)
      s->data->step = deltaS.segment<9>(9 * s->idx);
#endif

  }

  void EnergyFunctional::makeIDX() {
    for (unsigned int idx = 0; idx < frames.size(); idx++)
      frames[idx]->idx = idx;

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    for (unsigned int idx = 0; idx < speedAndBiases.size(); idx++)
      speedAndBiases[idx]->idx = idx;
#endif

    allPoints.clear();

    for (EFFrame *f : frames)
      for (EFPoint *p : f->points) {
        allPoints.push_back(p);
        for (EFResidual *r : p->residualsAll) {
          r->hostIDX = r->host->idx;
          if (r->data->staticStereo) //- static stereo residual
            r->targetIDX = -1;
          else
            r->targetIDX = r->target->idx;
        }
      }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)
    for (EFSpeedAndBias *s : speedAndBiases) {
      assert(s->idx == s->data->host->efFrame->idx);
      for (EFIMUResidual *r : s->residualsAll) {
        r->fromSBIDX = r->from_sb->idx;
        r->toSBIDX = r->to_sb->idx;
      }
    }
#endif


    EFIndicesValid = true;
  }


  VecX EnergyFunctional::getStitchedDeltaF() const {
#if defined(STEREO_MODE)
    VecX d = VecX(CPARS + nFrames * 10);
    d.head<CPARS>() = cDeltaF.cast<double>();
    for (int h = 0; h < nFrames; h++) d.segment<10>(CPARS + 10 * h) = frames[h]->delta;
    return d;
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    VecX d = VecX(CPARS + nFrames * 8);
    d.head<CPARS>() = cDeltaF.cast<double>();
    for (int h = 0; h < nFrames; h++) d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
    return d;
#endif
  }


}
