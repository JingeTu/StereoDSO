//
// Created by jg on 18-4-12.
//
#include "OptimizationBackend/PREEnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "FullSystem/CoarseTracker.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  bool PRE_EFAdjointsValid = false;
  bool PRE_EFIndicesValid = false;
  bool PRE_EFDeltaValid = false;

  void PREEnergyFunctional::setAdjointsF() {

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
        Vec2f affLL_r = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(),
                                                    target->aff_g2l_r_0()).cast<float>();
        if (t == h) {
          affLL = AffLight::fromToVecExposure(host->ab_exposure, host->rightFrame->ab_exposure, host->aff_g2l_0(),
                                              host->aff_g2l_r_0()).cast<float>();
        }

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
          AT(7, 7) = 1;
          AT(8, 8) = affLL_r[0];
          AT(9, 9) = 1;

          AH(6, 6) = -affLL[0];
          AH(7, 7) = -affLL[0];
          AH(8, 8) = 0;
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


    PRE_EFAdjointsValid = true;
  }

  PREEnergyFunctional::PREEnergyFunctional() {
    adHost = 0;
    adTarget = 0;

    red = 0;

    adHostF = 0;
    adTargetF = 0;
    adHTdeltaF = 0;

    nFrames = 0;
    nSpeedAndBiases = nIMUResiduals = nMargSpeedAndBiases = 0;

    resInA = resInL = resInM = 0;
    currentLambda = 0;
  }

  PREEnergyFunctional::~PREEnergyFunctional() {
    for (EFFrame *f : frames)
      delete f;

    for (EFSpeedAndBias *s : speedAndBiases) {
      s->data->efSB = 0;
      delete s;
    }

    if (adHost != 0) delete[] adHost;
    if (adTarget != 0) delete[] adTarget;


    if (adHostF != 0) delete[] adHostF;
    if (adTargetF != 0) delete[] adTargetF;
    if (adHTdeltaF != 0) delete[] adHTdeltaF;
  }

  void PREEnergyFunctional::clear() {
    HM = MatXX::Zero(nFrames * 10, nFrames * 10);
    bM = VecX::Zero(nFrames * 10);
  }

  void PREEnergyFunctional::setDeltaF() {
    if (adHTdeltaF != 0) delete[] adHTdeltaF;
    adHTdeltaF = new Mat110f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
      for (int t = 0; t < nFrames; t++) {
        int idx = h + t * nFrames;
        adHTdeltaF[idx] =
            frames[h]->data->get_state_minus_stateZero().head<10>().cast<float>().transpose() * adHostF[idx]
            + frames[t]->data->get_state_minus_stateZero().head<10>().cast<float>().transpose() * adTargetF[idx];
      }

    for (EFFrame *f : frames) {
      f->delta = f->data->get_state_minus_stateZero().head<10>();
      f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<10>();
    }

    PRE_EFDeltaValid = true;
  }

  void PREEnergyFunctional::accumulateIMUAF_MT(MatXX &H, VecX &b, bool MT) {
    // Use one thread first.
    H = MatXX::Zero(nFrames * 10, nFrames * 10);
    b = VecX::Zero(nFrames * 10);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (r->isLinearized) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = J->resF;


        int fIdx = 10 * r->from_f->idx;
        int tIdx = 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();
        if (!r->from_f->data->isKF || setting_PREnofixKF) {
          H.block<6, 6>(tIdx, fIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
          H.block<6, 6>(fIdx, fIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        }
        b.segment<6>(fIdx).noalias() += (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() += (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void PREEnergyFunctional::accumulateIMULF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10, nFrames * 10);
    b = VecX::Zero(nFrames * 10);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (!r->isLinearized) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = r->res_toZeroF;

        int fIdx = 10 * r->from_f->idx;
        int tIdx = 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();
        if (!r->from_f->data->isKF || setting_PREnofixKF) {
          H.block<6, 6>(tIdx, fIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
          H.block<6, 6>(fIdx, fIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        }
        b.segment<6>(fIdx).noalias() += (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() += (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void PREEnergyFunctional::accumulateIMUSCF_MT(MatXX &H, VecX &b, MatXX &Hss_inv, MatXX &Hsx, VecX &bsr, bool MT) {
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

        Hxsf.block<6, 9>(tIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[1];
        Hxsf.block<6, 9>(tIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[0];

        if (!r->from_f->data->isKF || setting_PREnofixKF) {
          Hxsf.block<6, 9>(fIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[0];
          Hxsf.block<6, 9>(fIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[1];
        }
        Eigen::Matrix<float, 15, 1> resApprox;

        if (r->isLinearized)
          resApprox = r->res_toZeroF;
        else
          resApprox = J->resF;

        bsrf.segment<9>(fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * resApprox;
        bsrf.segment<9>(tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * resApprox;
      }
    }

    H = MatXX::Zero(nFrames * 10, nFrames * 10);
    b = VecX::Zero(nFrames * 10);

    MatXXf Hssf_inv;
    if (nFrames == 3) {
      Hssf_inv = MatXXf::Zero(nFrames * 9, nFrames * 9);
      Hssf_inv.block(9, 9, 18, 18) = Hssf.block(9, 9, 18, 18).inverse();
    }
    else {
      Hssf_inv = Hssf.inverse();
    }

    MatXXf H_small = Hxsf * Hssf_inv * Hxsf.transpose(); //- nFrame * 6
    VecXf b_small = Hxsf * Hssf_inv * bsrf; //- nFrame * 6

    //- save result
    for (int r = 0; r < nFrames; r++) {
      int rIdx = 10 * r;
      H.block<6, 6>(rIdx, rIdx).noalias() = H_small.block<6, 6>(r * 6, r * 6).cast<double>();
      b.segment<6>(rIdx).noalias() = b_small.segment<6>(r * 6).cast<double>();
      for (int c = r + 1; c < nFrames; c++) {
        int cIdx = 10 * c;

        H.block<6, 6>(rIdx, cIdx).noalias() = H_small.block<6, 6>(r * 6, c * 6).cast<double>();
        H.block<6, 6>(cIdx, rIdx).noalias() = H_small.block<6, 6>(c * 6, r * 6).cast<double>();
      }
    }
    Hss_inv = Hssf_inv.cast<double>();
    Hsx = Hxsf.transpose().cast<double>();
    bsr = bsrf.cast<double>();
  }

  void PREEnergyFunctional::accumulateIMUMF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10, nFrames * 10);
    b = VecX::Zero(nFrames * 10);

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        if (!r->flaggedForMarginalization) continue;
        RawIMUResidualJacobian *J = r->J;

        Eigen::Matrix<float, 15, 1> resApprox = r->res_toZeroF;

        int fIdx = 10 * r->from_f->idx;
        int tIdx = 10 * r->to_f->idx;
        H.block<6, 6>(fIdx, tIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[1]).cast<double>();
        H.block<6, 6>(tIdx, tIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[1]).cast<double>();

        if (!r->from_f->data->isKF || setting_PREnofixKF) {
          H.block<6, 6>(tIdx, fIdx).noalias() += (J->Jrdxi[1].transpose() * J->Jrdxi[0]).cast<double>();
          H.block<6, 6>(fIdx, fIdx).noalias() += (J->Jrdxi[0].transpose() * J->Jrdxi[0]).cast<double>();
        }

        b.segment<6>(fIdx).noalias() += (J->Jrdxi[0].transpose() * resApprox).cast<double>();
        b.segment<6>(tIdx).noalias() += (J->Jrdxi[1].transpose() * resApprox).cast<double>();
      }
    }
  }

  void PREEnergyFunctional::accumulateIMUMSCF_MT(MatXX &H, VecX &b, bool MT) {
    H = MatXX::Zero(nFrames * 10, nFrames * 10);
    b = VecX::Zero(nFrames * 10);

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

        if (fIdxRaw != -1 && tIdxRaw != -1) {
          pair[fIdxRaw] = r->from_f->idx;
          pair[tIdxRaw] = r->to_f->idx;
          Hssf.block<9, 9>(fIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[0];
          Hssf.block<9, 9>(tIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[1];
          Hssf.block<9, 9>(fIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[1];
          Hssf.block<9, 9>(tIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[0];

          Hxsf.block<6, 9>(fIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[0];
          Hxsf.block<6, 9>(tIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[1];
          Hxsf.block<6, 9>(fIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[1];
          Hxsf.block<6, 9>(tIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[0];
        }
        else if (fIdxRaw == -1 && tIdxRaw != -1) {
          pair[tIdxRaw] = r->to_f->idx;
          Hssf.block<9, 9>(tIdxRaw * 9, tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * J->Jrdsb[1];
          Hxsf.block<6, 9>(tIdxRaw * 6, tIdxRaw * 9).noalias() += J->Jrdxi[1].transpose() * J->Jrdsb[1];
        }
        else if (fIdxRaw != -1 && tIdxRaw == -1) {
          pair[fIdxRaw] = r->from_f->idx;
          Hssf.block<9, 9>(fIdxRaw * 9, fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * J->Jrdsb[0];
          Hxsf.block<6, 9>(fIdxRaw * 6, fIdxRaw * 9).noalias() += J->Jrdxi[0].transpose() * J->Jrdsb[0];
        }

        Eigen::Matrix<float, 15, 1> resApprox;

        if (r->isLinearized)
          resApprox = r->res_toZeroF;
        else
          resApprox = J->resF;

        if (fIdxRaw != -1)
          bsrf.segment<9>(fIdxRaw * 9).noalias() += J->Jrdsb[0].transpose() * resApprox;
        if (tIdxRaw != -1)
          bsrf.segment<9>(tIdxRaw * 9).noalias() += J->Jrdsb[1].transpose() * resApprox;
      }
    }

    MatXXf Hssf_inv = Hssf.inverse();//- This inverse is unavoidable for me now, unless fix one speedAndBias point.

    MatXXf H_small = Hxsf * Hssf_inv * Hxsf.transpose(); //- nFrame * 6
    VecXf b_small = Hxsf * Hssf_inv * bsrf; //- nFrame * 6

    //- save result
    for (int r = 0; r < nMargSpeedAndBiases; r++) {
      int rBIdx = 10 * pair[r];
      H.block<6, 6>(rBIdx, rBIdx).noalias() = H_small.block<6, 6>(r * 6, r * 6).cast<double>();
      b.segment<6>(rBIdx).noalias() = b_small.segment<6>(r * 6).cast<double>();
      for (int c = r + 1; c < nMargSpeedAndBiases; c++) {
        int cBIdx = 10 * pair[c];

        H.block<6, 6>(rBIdx, cBIdx).noalias() = H_small.block<6, 6>(r * 6, c * 6).cast<double>();
        H.block<6, 6>(cBIdx, rBIdx).noalias() = H_small.block<6, 6>(c * 6, r * 6).cast<double>();
      }
    }

    delete[] pair;
  }

  void PREEnergyFunctional::resubstituteF_MT(VecX x, bool MT) {
    assert(x.size() == nFrames * 10);
    VecXf xF = x.cast<float>();

    Mat110f *xAd = new Mat110f[nFrames * nFrames];
    for (EFFrame *h : frames) {
      h->data->step = x.segment<10>(10 * h->idx);

      for (EFFrame *t : frames)
        xAd[nFrames * h->idx + t->idx] =
            xF.segment<10>(10 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx]
            + xF.segment<10>(10 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }

    delete[] xAd;
  }

  double PREEnergyFunctional::calcMEnergyF() {

    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);

    VecX delta = getStitchedDeltaF();
    return delta.dot(2 * bM + HM * delta);
  }

  double PREEnergyFunctional::calcLEnergyF_MT() {
    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);

    double E = 0;
    for (EFFrame *f : frames)
      E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

    return E;
  }

  EFIMUResidual *PREEnergyFunctional::insertIMUResidual(IMUResidual *r) {
    EFIMUResidual *efr = new EFIMUResidual(r, r->from_sb->efSB, r->to_sb->efSB, r->from_sb->host->PRE_efFrame,
                                           r->to_sb->host->PRE_efFrame);
    efr->idxInAll = r->to_sb->efSB->residualsAll.size();
    r->to_sb->efSB->residualsAll.push_back(efr); //- toSpeedAndBias as host.

    nIMUResiduals++;
    r->efIMUResidual = efr;
    return efr;
  }

  EFSpeedAndBias *PREEnergyFunctional::insertSpeedAndBias(SpeedAndBiasHessian *sh) {
    EFSpeedAndBias *efs = new EFSpeedAndBias(sh);
    efs->idx = speedAndBiases.size();
    speedAndBiases.push_back(efs);

    nSpeedAndBiases++;
    sh->efSB = efs;

    makeIDX();
    return efs;
  }

  EFFrame *PREEnergyFunctional::insertFrame(FrameHessian *fh) {
    EFFrame *eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff);

    nFrames++;
    fh->PRE_efFrame = eff;

    assert(HM.cols() == 10 * nFrames - 10);
    bM.conservativeResize(10 * nFrames);
    HM.conservativeResize(10 * nFrames, 10 * nFrames);
    bM.tail<10>().setZero();
    HM.rightCols<10>().setZero();
    HM.bottomRows<10>().setZero();
    PRE_EFIndicesValid = false;
    PRE_EFAdjointsValid = false;
    PRE_EFDeltaValid = false;

    setAdjointsF();
    makeIDX();

    return eff;
  }

  void PREEnergyFunctional::dropFrame(EFFrame *efF) {
    for (unsigned int i = efF->idx; i + 1 < frames.size(); i++) {
      frames[i] = frames[i + 1];
      frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    efF->data->PRE_efFrame = 0;
  }

  void PREEnergyFunctional::dropIMUResidual(EFIMUResidual *r) {
    EFSpeedAndBias *s = r->to_sb;
    assert(r == s->residualsAll[r->idxInAll]);

    s->residualsAll[r->idxInAll] = s->residualsAll.back();
    s->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    s->residualsAll.pop_back();

    nIMUResiduals--;
    r->data->efIMUResidual = 0;
    delete r;
  }

  void PREEnergyFunctional::marginalizeFrame(EFFrame *efF) {

    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);


    int ndim = nFrames * 10 - 10;// new dimension
    int odim = nFrames * 10;// old dimension

    if ((int) efF->idx != (int) frames.size() - 1) {
      int io = efF->idx * 10;  // index of frame to move to end
      int ntail = 10 * (nFrames - efF->idx - 1);
      assert((io + 10 + ntail) == nFrames * 10);

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
      hpi = Mat1010::Zero();
      Mat66 botRht66;
      botRht66 = HMScaled.bottomRightCorner<10, 10>().topLeftCorner<6, 6>();
      if (botRht66.norm() != 0) {
        botRht66 = 0.5f * (botRht66 + botRht66);
        botRht66 = botRht66.inverse();
        botRht66 = 0.5f * (botRht66 + botRht66);
        hpi.block<6, 6>(0, 0) = botRht66;
      }
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
    efF->data->PRE_efFrame = 0;

    assert((int) frames.size() * 10 == (int) HM.rows());
    assert((int) frames.size() * 10 == (int) HM.cols());
    assert((int) frames.size() * 10 == (int) bM.size());
    assert((int) frames.size() == (int) nFrames);

//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

    PRE_EFIndicesValid = false;
    PRE_EFAdjointsValid = false;
    PRE_EFDeltaValid = false;

    makeIDX();
    delete efF;
  }

  void PREEnergyFunctional::marginalizePointsF(const Mat1010 &M_last, const Vec10 &Mb_last,
                                               const Mat1010 &Msc_last, const Vec10 &Mbsc_last) {
    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);

    // adHost adTarget
    MatXX H = MatXX::Zero(nFrames * 10, nFrames * 10);
    VecX b = VecX::Zero(nFrames * 10);

    Mat1010 M = M_last - Msc_last;
    Vec10 Mb = Mb_last - Mbsc_last;

    {
      int h = 0;
      int t = nFrames - 1;
      int aidx = h + nFrames * t;
      H.block<10, 10>(t * 10, t * 10).noalias() += adTarget[aidx] * M * adTarget[aidx].transpose();
      H.block<10, 10>(h * 10, t * 10).noalias() += adHost[aidx] * M * adTarget[aidx].transpose();
      if (setting_PREnofixKF) {
        H.block<10, 10>(h * 10, h * 10).noalias() += adHost[aidx] * M * adHost[aidx].transpose();
        H.block<10, 10>(t * 10, h * 10).noalias() += adTarget[aidx] * M * adHost[aidx].transpose();
      }
      b.segment<10>(h * 10).noalias() += adHost[aidx].transpose() * Mb;
      b.segment<10>(t * 10).noalias() += adTarget[aidx].transpose() * Mb;
    }

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
      // have a look if prior is there.
      bool haveFirstFrame = false;
      for (EFFrame *f : frames) if (f->frameID == 0) haveFirstFrame = true;

      if (!haveFirstFrame)
        orthogonalize(&b, &H);
    }

    HM += setting_margWeightFac * H;
    bM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL) {
      if (setting_PREnofixKF) {
        orthogonalize(&bM, &HM);
      }
      else {
        MatXX HM_temp = HM.block(10, 10, nFrames * 10 - 10, nFrames * 10 - 10);
        VecX bM_temp = bM.segment(10, nFrames * 10 - 10);
        orthogonalize(&bM_temp, &HM_temp);
        HM.block(10, 10, nFrames * 10 - 10, nFrames * 10 - 10) = HM_temp;
        bM.segment(10, nFrames * 10 - 10) = bM_temp;
      }
    }

    PRE_EFIndicesValid = false;
    makeIDX();
  }

  void PREEnergyFunctional::marginalizeSpeedAndBiasesF() {
    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);

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

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL) {
      MatXX HM_temp = HM.block(10, 10, nFrames * 10 - 10, nFrames * 10 - 10);
      VecX bM_temp = bM.segment(10, nFrames * 10 - 10);
      orthogonalize(&bM_temp, &HM_temp);
      HM.block(10, 10, nFrames * 10 - 10, nFrames * 10 - 10) = HM_temp;
      bM.segment(10, nFrames * 10 - 10) = bM_temp;
    }

    for (unsigned int i = 0; i < speedAndBiases.size(); i++) {
      EFSpeedAndBias *s = speedAndBiases[i];
      for (unsigned int j = 0; j < s->residualsAll.size(); j++) {
        EFIMUResidual *r = s->residualsAll[j];
        if (r->flaggedForMarginalization) {
          r->data->efIMUResidual = 0;
          dropIMUResidual(r);
          j--;
        }
      }
    }

    for (unsigned int i = 0; i < speedAndBiases.size(); i++) {
      EFSpeedAndBias *s = speedAndBiases[i];
      if (s->data->PRE_flaggedForMarginalization) {
        s->data->efSB = 0;
        deleteOutOrder<EFSpeedAndBias>(speedAndBiases, s);
        i--;
      }
    }

    PRE_EFIndicesValid = false;
    makeIDX();
  }

  void PREEnergyFunctional::orthogonalize(VecX *b, MatXX *H) {
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


  void PREEnergyFunctional::solveSystemF(int iteration, double lambda, Mat1010 &H_last, Vec10 &b_last) {
    if (setting_solverMode & SOLVER_USE_GN) lambda = 0;
    if (setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

    assert(PRE_EFDeltaValid);
    assert(PRE_EFAdjointsValid);
    assert(PRE_EFIndicesValid);

    MatXX HL_top, HA_top, H_sc;
    VecX bL_top, bA_top, bM_top, b_sc;

    HL_top = MatXX::Zero(nFrames * 10, nFrames * 10);
    HA_top = MatXX::Zero(nFrames * 10, nFrames * 10);
    H_sc = MatXX::Zero(nFrames * 10, nFrames * 10);

    bL_top = VecX::Zero(nFrames * 10);
    bA_top = VecX::Zero(nFrames * 10);
    bM_top = VecX::Zero(nFrames * 10);
    b_sc = VecX::Zero(nFrames * 10);

    {
      int h = 0;
      int t = nFrames - 1;
      int aidx = h + nFrames * t;
      HA_top.block<10, 10>(t * 10, t * 10).noalias() += adTarget[aidx] * H_last * adTarget[aidx].transpose();
      HA_top.block<10, 10>(h * 10, t * 10).noalias() += adHost[aidx] * H_last * adTarget[aidx].transpose();
      if (setting_PREnofixKF) {
        HA_top.block<10, 10>(h * 10, h * 10).noalias() += adHost[aidx] * H_last * adHost[aidx].transpose();
        HA_top.block<10, 10>(t * 10, h * 10).noalias() += adTarget[aidx] * H_last * adHost[aidx].transpose();
      }
      bA_top.segment<10>(h * 10).noalias() += adHost[aidx].transpose() * b_last;
      bA_top.segment<10>(t * 10).noalias() += adTarget[aidx].transpose() * b_last;
    }

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
      for (int i = 0; i < 10 * nFrames; i++) HFinal_top(i, i) *= (1 + lambda);
    }
    else { //- here


      HFinal_top = HL_top + HM + HA_top;
      bFinal_top = bL_top + bM_top + bA_top - b_sc;

      lastHS = HFinal_top - H_sc;
      lastbS = bFinal_top;

      for (int i = 0; i < 10 * nFrames; i++) HFinal_top(i, i) *= (1 + lambda);

      HFinal_top -= H_sc * (1.0f / (1 + lambda));
    }

    if (!setting_PREnofixKF) {
      int solveSize = 10 * nFrames - 10;

      HFinal_top = HFinal_top.block(10, 10, solveSize, solveSize);
      bFinal_top = bFinal_top.segment(10, solveSize);
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
    }

    if (!std::isfinite(x.norm())) {
//      LOG(INFO) << "HM: " << HM;
//      LOG(INFO) << "bM: " << bM;
      LOG(INFO) << "HFinal_top: " << HFinal_top;
      LOG(INFO) << "bFinal_top: " << bFinal_top.transpose();
//      LOG(INFO) << "HA_top_imu: " << HA_top_imu;
//      LOG(INFO) << "H_sc_imu: " << H_sc_imu;
      LOG(INFO) << "x: " << x.transpose();
      LOG(INFO) << "nFrames: " << nFrames;
      assert(std::isfinite(x.norm()));
    }

    if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
        (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
      VecX xOld = x;
      orthogonalize(&x, 0);
    }

    lastX = x;

    if (!setting_PREnofixKF) {
      VecX newx = VecX::Zero(nFrames * 10);
      newx.segment(10, nFrames * 10 - 10) = x;
      x = newx;
    }

    currentLambda = lambda;
    //- Modify step.
    resubstituteF_MT(x, multiThreading);
    currentLambda = 0;
    //- resubstitute speedAndBiases
    //- Prepare \delta_X first
    VecX deltaX = VecX::Zero(nFrames * 6);
    for (int r = 0; r < nFrames; r++) {
      int rIdx = r * 10;
      deltaX.segment<6>(r * 6) = x.segment<6>(rIdx);
    }
    VecX deltaS = -Hss_inv * (bsr + Hsx * deltaX);
    for (EFSpeedAndBias *s : speedAndBiases)
      s->data->step = deltaS.segment<9>(9 * s->data->idx);
    LOG(INFO) << "x: " << x.transpose();
    LOG(INFO) << "deltaS: " << deltaS.transpose();
  }

  void PREEnergyFunctional::makeIDX() {
    for (unsigned int idx = 0; idx < frames.size(); idx++)
      frames[idx]->idx = idx;

    for (unsigned int idx = 0; idx < speedAndBiases.size(); idx++)
      speedAndBiases[idx]->idx = idx;

    for (EFSpeedAndBias *s : speedAndBiases) {
      for (EFIMUResidual *r : s->residualsAll) {
        r->fromSBIDX = r->from_sb->idx;
        r->toSBIDX = r->to_sb->idx;
      }
    }

    PRE_EFIndicesValid = true;
  }


  VecX PREEnergyFunctional::getStitchedDeltaF() const {
    VecX d = VecX(nFrames * 10);
    for (int h = 0; h < nFrames; h++) d.segment<10>(10 * h) = frames[h]->delta;
    return d;
  }

#endif
}