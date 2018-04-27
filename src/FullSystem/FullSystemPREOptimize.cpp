//
// Created by jg on 18-4-8.
//

#include "FullSystem/FullSystem.h"
#include "FullSystem/CoarseTracker.h"
#include "OptimizationBackend/PREEnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso {
#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  void FullSystem::PRE_optimize(int mnumOptIts) {
    if (PRE_frameHessians.size() < 2) return; //- if there is only 1 frame just skip
    if (PRE_frameHessians.size() < 3) mnumOptIts = 20;
    if (PRE_frameHessians.size() < 4) mnumOptIts = 15;
    assert(PRE_frameHessians.size() < 4);
    assert(coarseTracker->lastRef == PRE_frameHessians.front());

    PRE_ef->makeIDX();
    PRE_activeIMUResiduals.clear();
    for (SpeedAndBiasHessian *sh : PRE_speedAndBiasHessians) {
      for (IMUResidual *r : sh->residuals)
        if (!r->efIMUResidual->isLinearized)
          PRE_activeIMUResiduals.push_back(r);
    }

    Vec3 lastEnergy = PRE_linearizeAll(false);
    double lastEnergyL = PRE_calcLEnergy();
    double lastEnergyM = PRE_calcMEnergy();

    if (multiThreading)
      treadReduce.reduce(boost::bind(&FullSystem::PRE_applyIMURes_Reductor, this, true, _1, _2, _3, _4), 0,
                         PRE_activeIMUResiduals.size(), 50);
    else
      PRE_applyIMURes_Reductor(true, 0, PRE_activeIMUResiduals.size(), 0, 0);

    if (!setting_debugout_runquiet) {
      LOG(INFO) << "Initial Error       \t";
      printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, PRE_frameHessians.back()->aff_g2l().a,
                  PRE_frameHessians.back()->aff_g2l().b);
    }

    double lambda = 1e-1;
    float stepsize = 1;
    VecX previousX;
    if (setting_PREnofixKF) {
      previousX = VecX::Constant(10 * PRE_frameHessians.size(), NAN);
    }
    else {
      previousX = VecX::Constant(10 * PRE_frameHessians.size() - 10, NAN);
    }

    for (int iteration = 0; iteration < mnumOptIts; iteration++) {
      PRE_backupState(iteration != 0);
      PRE_solveSystem(iteration, lambda);
      double incDirChange = (1e-20 + previousX.dot(PRE_ef->lastX)) / (1e-20 + previousX.norm() * PRE_ef->lastX.norm());
      previousX = PRE_ef->lastX;

      if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
        float newStepsize = exp(incDirChange * 1.4);
        if (incDirChange < 0 && stepsize > 1) stepsize = 1;

        stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
        if (stepsize > 2) stepsize = 2;
        if (stepsize < 0.25) stepsize = 0.25;
      }

      bool canbreak = PRE_doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

      Vec3 newEnergy = PRE_linearizeAll(false);
      double newEnergyL = calcLEnergy();
      double newEnergyM = calcMEnergy();


      if (!setting_debugout_runquiet) {
        char buf[256];
        sprintf(buf, "%s %d (L %.2f, dir %.2f, ss %.1f): \t",
                (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                 lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
                iteration,
                log10(lambda),
                incDirChange,
                stepsize);
        LOG(INFO) << buf;
        printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, PRE_frameHessians.back()->aff_g2l().a,
                    PRE_frameHessians.back()->aff_g2l().b);
      }

      if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] <
                                     lastEnergy[0] + lastEnergy[1])) {

        if (multiThreading)
          treadReduce.reduce(boost::bind(&FullSystem::PRE_applyIMURes_Reductor, this, true, _1, _2, _3, _4), 0,
                             PRE_activeIMUResiduals.size(), 50);
        else
          PRE_applyIMURes_Reductor(true, 0, PRE_activeIMUResiduals.size(), 0, 0);

        lastEnergy = newEnergy;
        lastEnergyL = newEnergyL;
        lastEnergyM = newEnergyM;

        lambda *= 0.25;
      }
      else {
        PRE_loadSateBackup();
        lastEnergy = PRE_linearizeAll(false);
        lastEnergyL = calcLEnergy();
        lastEnergyM = calcMEnergy();
        lambda *= 1e2;
      }

      if (canbreak && iteration >= setting_minOptIterations) break;
    }


    Vec10 newStateZero = Vec10::Zero();
    newStateZero.segment<4>(6) = PRE_frameHessians.back()->get_state().segment<4>(6);
    PRE_frameHessians.back()->setEvalPT(PRE_frameHessians.back()->PRE_T_CW,
                                        newStateZero);
    PRE_EFDeltaValid = false;
    PRE_EFAdjointsValid = false;
    PRE_ef->setAdjointsF();
    PRE_setPrecalcValues();

    lastEnergy = PRE_linearizeAll(true);

    {

      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      for (FrameHessian *fh : PRE_frameHessians) {
        fh->shell->T_WC = fh->PRE_T_WC;
        fh->shell->aff_g2l = fh->aff_g2l();
        fh->rightFrame->shell->aff_g2l = fh->aff_g2l_r();
      }
    }
  }

  void FullSystem::PRE_setPrecalcValues() {
    PRE_ef->setDeltaF();
  }

  Vec3 FullSystem::PRE_linearizeAll(bool fixLinearization) {
    double lastEnergyP = 0;
    double lastEnergyR = 0;
    double num = 0;

    //- 1. vision
    //- Use coarseTracker to calculate vision residual
    FrameHessian *lastfh = PRE_frameHessians.back();
    Vec6 res = coarseTracker->calculateRes(lastfh, lastfh->rightFrame);
    lastEnergyP += res[0];

    //- 2. imu
    if (multiThreading) {
      treadReduce.reduce(
          boost::bind(&FullSystem::PRE_linearizeAllIMU_Reductor, this, fixLinearization, _1, _2, _3, _4), 0,
          PRE_activeIMUResiduals.size(), 0);
      lastEnergyP += treadReduce.stats[0];
    }
    else {
      Vec10 stats;
      stats.setZero();
      PRE_linearizeAllIMU_Reductor(fixLinearization, 0, PRE_activeIMUResiduals.size(), &stats, 0);
      LOG(INFO) << "stats[0]: " << stats[0];
      lastEnergyP += stats[0];
    }

    return Vec3(lastEnergyP, lastEnergyR, num);
  }

  void FullSystem::PRE_linearizeAllIMU_Reductor(bool fixLinearization, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++) {
      IMUResidual *r = PRE_activeIMUResiduals[k];
      (*stats)[0] += r->linearize(&imuParameters);
      if (fixLinearization)
        r->applyRes(true);
    }
  }

  void FullSystem::PRE_applyIMURes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid) {
    for (int k = min; k < max; k++)
      PRE_activeIMUResiduals[k]->applyRes(true);
  }

  void FullSystem::PRE_backupState(bool backupLastStep) {
    if (setting_solverMode & SOLVER_MOMENTUM) {
      if (backupLastStep) {
        for (FrameHessian *fh : PRE_frameHessians) {
          fh->step_backup = fh->step;
          fh->state_backup = fh->get_state();
        }
        for (SpeedAndBiasHessian *sh : PRE_speedAndBiasHessians) {
          sh->step_backup = sh->step;
          sh->state_backup = sh->get_state();
        }
      }
      else {
        for (FrameHessian *fh : PRE_frameHessians) {
          fh->step_backup.setZero();
          fh->state_backup = fh->get_state();
        }
        for (SpeedAndBiasHessian *sh : PRE_speedAndBiasHessians) {
          sh->step_backup.setZero();
          sh->state_backup = sh->get_state();
        }
      }
    }
    else {
      for (FrameHessian *fh : PRE_frameHessians) {
        fh->state_backup = fh->get_state();
      }
      for (SpeedAndBiasHessian *sh : PRE_speedAndBiasHessians) {
        sh->state_backup = sh->get_state();
      }
    }

    PRE_setPrecalcValues();
  }

  void FullSystem::PRE_loadSateBackup() {
    for (FrameHessian *fh : PRE_frameHessians)
      fh->setState(fh->state_backup);
    for (SpeedAndBiasHessian *sh : PRE_speedAndBiasHessians)
      sh->setState(sh->state_backup);

    PRE_EFDeltaValid = false;
    PRE_setPrecalcValues();
  }

  void FullSystem::PRE_solveSystem(int iteration, double lambda) {
    PRE_getNullspaces(
        PRE_ef->lastNullspaces_pose,
        PRE_ef->lastNullspaces_scale,
        PRE_ef->lastNullspaces_affA,
        PRE_ef->lastNullspaces_affB);
    FrameHessian *lastfh = PRE_frameHessians.back();
    Mat1010 H;
    Vec10 b;
    coarseTracker->calculateHAndb(lastfh, lastfh->rightFrame, H, b);
    PRE_ef->solveSystemF(iteration, lambda, H, b);
  }

  bool FullSystem::PRE_doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA,
                                        float stepfacD) {
    Vec10 pstepfac;
    pstepfac.segment<3>(0).setConstant(stepfacT);
    pstepfac.segment<3>(3).setConstant(stepfacR);
    pstepfac.segment<4>(6).setConstant(stepfacA);

    float sumA = 0, sumB = 0, sumR = 0;

    if (setting_solverMode & SOLVER_MOMENTUM) {
      for (FrameHessian *fh : PRE_frameHessians) {
        Vec10 step = fh->step;
        step.head<6>() += 0.5f * (fh->step_backup.head<6>());

        fh->setState(fh->state_backup + step);
        sumA += step[6] * step[6];
        sumB += step[7] * step[7];
        sumR += step.segment<3>(3).squaredNorm();
      }
    }
    else {
      for (FrameHessian *fh : PRE_frameHessians) {
        fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));

        sumA += fh->step[6] * fh->step[6];
        sumB += fh->step[7] * fh->step[7];
        sumR += fh->step.segment<3>(3).squaredNorm();
      }
    }

    sumA /= PRE_frameHessians.size();
    sumB /= PRE_frameHessians.size();
    sumR /= PRE_frameHessians.size();

    if (!setting_debugout_runquiet) {
      char buf[256];
      sprintf(buf, "STEPS: A %.1f; B %.1f; R %.1f. \t",
              sqrtf(sumA) / (0.0005 * setting_thOptIterations),
              sqrtf(sumB) / (0.00005 * setting_thOptIterations),
              sqrtf(sumR) / (0.00005 * setting_thOptIterations));
      LOG(INFO) << buf;
    }

    PRE_EFDeltaValid = false;
    PRE_setPrecalcValues();

    return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
           sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
           sqrtf(sumR) < 0.00005 * setting_thOptIterations;
  }

  void FullSystem::PRE_flagFramesForMarginalization() {

    assert(PRE_frameHessians.size() <= 3);

    if (PRE_frameHessians.size() == 2) {
//      PRE_frameHessians[0]->speedAndBiasHessian->PRE_flaggedForMarginalization = true;
    }

    if (PRE_frameHessians.size() == 3) {
      PRE_frameHessians[1]->PRE_flaggedForMarginalization = true;
//      PRE_frameHessians[1]->speedAndBiasHessian->PRE_flaggedForMarginalization = true;
    }
  }

  void FullSystem::PRE_flagIMUResidualsForRemoval() {
    for (SpeedAndBiasHessian *s : PRE_speedAndBiasHessians) {
      for (IMUResidual *r : s->residuals) {
        if (r->from_sb->PRE_flaggedForMarginalization || r->to_sb->PRE_flaggedForMarginalization) {
          r->efIMUResidual->isLinearized = false;
          r->efIMUResidual->flaggedForMarginalization = true;
        }
      }
    }
  }

  void FullSystem::PRE_makeSpeedAndBiasesMargIDXForMarginalization() {
    int count = 0;
    for (SpeedAndBiasHessian *s : PRE_speedAndBiasHessians) {
      if (s->PRE_flaggedForMarginalization) s->margIDX = count++;
    }
    PRE_ef->nMargSpeedAndBiases = count;
  }

  void FullSystem::PRE_marginalizeSpeedAndBiases() {
    PRE_ef->marginalizeSpeedAndBiasesF();

    for (unsigned int i = 0; i < PRE_speedAndBiasHessians.size(); i++) {
      SpeedAndBiasHessian *sh = PRE_speedAndBiasHessians[i];
      for (unsigned int j = 0; j < sh->residuals.size(); j++) {
        IMUResidual *r = sh->residuals[j];
        if (r->efIMUResidual == 0) {
          deleteOutOrder<IMUResidual>(sh->residuals, j);
          j--;
        }
      }
      if (sh->efSB == 0) {
        deleteOutOrder<SpeedAndBiasHessian>(PRE_speedAndBiasHessians, sh);
        i--;
      }
    }
  }

  void FullSystem::PRE_marginalizePoints() {
    Mat1010 M, Msc;
    Vec10 b, bsc;
    FrameHessian *lastfh = PRE_frameHessians.back();
    coarseTracker->calculateHAndb(lastfh, lastfh->rightFrame, M, b);
    coarseTracker->calculateMscAndbsc(lastfh, lastfh->rightFrame, Msc, bsc);

    PRE_ef->marginalizePointsF(M, b, Msc, bsc);
  }

  void FullSystem::PRE_marginalizeFrame(FrameHessian *frame) {
    // marginalize or remove all this frames points.

    PRE_ef->marginalizeFrame(frame->PRE_efFrame);

    if (frame->isKF) {
      popOutOrder<FrameHessian>(PRE_frameHessians, frame);
      popOutOrder<FrameHessian>(PRE_frameHessiansRight, frame->rightFrame);
    }
    else {
      deleteOutOrder<FrameHessian>(PRE_frameHessians, frame);
      deleteOutOrder<FrameHessian>(PRE_frameHessiansRight, frame->rightFrame);
    }
    for (unsigned int i = 0; i < PRE_frameHessians.size(); i++)
      PRE_frameHessians[i]->PRE_idx = i;
    for (unsigned int i = 0; i < PRE_frameHessiansRight.size(); i++)
      PRE_frameHessiansRight[i]->PRE_idx = i;

    PRE_setPrecalcValues();
    PRE_ef->setAdjointsF();
  }

  std::vector<VecX> FullSystem::PRE_getNullspaces(
      std::vector<VecX> &nullspaces_pose,
      std::vector<VecX> &nullspaces_scale,
      std::vector<VecX> &nullspaces_affA,
      std::vector<VecX> &nullspaces_affB) {
    nullspaces_pose.clear();
    nullspaces_scale.clear();
    nullspaces_affA.clear();
    nullspaces_affB.clear();


    std::vector<VecX> nullspaces_x0_pre;
    if (setting_PREnofixKF) {
      int n = PRE_frameHessians.size() * 10;
      for (int i = 0; i < 6; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : PRE_frameHessians) {
          nullspace_x0.segment<6>(fh->PRE_idx * 10) = fh->nullspaces_pose.col(i);
          nullspace_x0.segment<3>(fh->PRE_idx * 10) *= SCALE_XI_TRANS_INVERSE;
          nullspace_x0.segment<3>(fh->PRE_idx * 10 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_pose.push_back(nullspace_x0);
      }
      for (int i = 0; i < 2; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : PRE_frameHessians) {
          nullspace_x0.segment<4>(fh->PRE_idx * 10 + 6) = fh->nullspaces_affine.col(i).head<4>();
          nullspace_x0[fh->PRE_idx * 10 + 6] *= SCALE_A_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 + 7] *= SCALE_B_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 + 8] *= SCALE_A_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 + 9] *= SCALE_B_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        if (i == 0) nullspaces_affA.push_back(nullspace_x0);
        if (i == 1) nullspaces_affB.push_back(nullspace_x0);
      }

      VecX nullspace_x0(n);
      nullspace_x0.setZero();
      for (FrameHessian *fh : PRE_frameHessians) {
        nullspace_x0.segment<6>(fh->PRE_idx * 10) = fh->nullspaces_scale;
        nullspace_x0.segment<3>(fh->PRE_idx * 10) *= SCALE_XI_TRANS_INVERSE;
        nullspace_x0.segment<3>(fh->PRE_idx * 10 + 3) *= SCALE_XI_ROT_INVERSE;
      }
      nullspaces_x0_pre.push_back(nullspace_x0);
      nullspaces_scale.push_back(nullspace_x0);
    }
    else {
      int n = PRE_frameHessians.size() * 10 - 10;
      for (int i = 0; i < 6; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : PRE_frameHessians) {
          if (fh->isKF) continue;
          nullspace_x0.segment<6>(fh->PRE_idx * 10 - 10) = fh->nullspaces_pose.col(i);
          nullspace_x0.segment<3>(fh->PRE_idx * 10 - 10) *= SCALE_XI_TRANS_INVERSE;
          nullspace_x0.segment<3>(fh->PRE_idx * 10 - 10 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_pose.push_back(nullspace_x0);
      }
      for (int i = 0; i < 2; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (FrameHessian *fh : PRE_frameHessians) {
          if (fh->isKF) continue;
          nullspace_x0.segment<4>(fh->PRE_idx * 10 - 10 + 6) = fh->nullspaces_affine.col(i).head<4>();
          nullspace_x0[fh->PRE_idx * 10 - 10 + 6] *= SCALE_A_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 - 10 + 7] *= SCALE_B_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 - 10 + 8] *= SCALE_A_INVERSE;
          nullspace_x0[fh->PRE_idx * 10 - 10 + 9] *= SCALE_B_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        if (i == 0) nullspaces_affA.push_back(nullspace_x0);
        if (i == 1) nullspaces_affB.push_back(nullspace_x0);
      }

      VecX nullspace_x0(n);
      nullspace_x0.setZero();
      for (FrameHessian *fh : PRE_frameHessians) {
        if (fh->isKF) continue;
        nullspace_x0.segment<6>(fh->PRE_idx * 10 - 10) = fh->nullspaces_scale;
        nullspace_x0.segment<3>(fh->PRE_idx * 10 - 10) *= SCALE_XI_TRANS_INVERSE;
        nullspace_x0.segment<3>(fh->PRE_idx * 10 - 10 + 3) *= SCALE_XI_ROT_INVERSE;
      }
      nullspaces_x0_pre.push_back(nullspace_x0);
      nullspaces_scale.push_back(nullspace_x0);
    }

    return nullspaces_x0_pre;
  }

  double FullSystem::PRE_calcLEnergy() {
    if (setting_forceAceptStep) return 0;

    double Ef = PRE_ef->calcLEnergyF_MT();
    return Ef;
  }

  double FullSystem::PRE_calcMEnergy() {
    if (setting_forceAceptStep) return 0;
    // calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
    //ef->makeIDX();
    //ef->setDeltaF(&Hcalib);
    return PRE_ef->calcMEnergyF();
  }

#endif
}