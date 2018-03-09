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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>
#include <cv.h>

namespace dso {
  int FrameHessian::instanceCounter = 0;
  int PointHessian::instanceCounter = 0;
  int CalibHessian::instanceCounter = 0;


  FullSystem::FullSystem() : leftToRight_SE3(SE3(Sophus::Quaterniond(1, 0, 0, 0), Vec3(-baseline, 0, 0))) {

//    std::cout << leftToRight_SE3.matrix3x4() << std::endl;
//
//    printf("baseline: %f\n", baseline);
    int retstat = 0;
    if (setting_logStuff) {

      retstat += system("rm -rf logs");
      retstat += system("mkdir logs");

      retstat += system("rm -rf mats");
      retstat += system("mkdir mats");

      calibLog = new std::ofstream();
      calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
      calibLog->precision(12);

      numsLog = new std::ofstream();
      numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
      numsLog->precision(10);

      coarseTrackingLog = new std::ofstream();
      coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
      coarseTrackingLog->precision(10);

      eigenAllLog = new std::ofstream();
      eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
      eigenAllLog->precision(10);

      eigenPLog = new std::ofstream();
      eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
      eigenPLog->precision(10);

      eigenALog = new std::ofstream();
      eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
      eigenALog->precision(10);

      DiagonalLog = new std::ofstream();
      DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
      DiagonalLog->precision(10);

      variancesLog = new std::ofstream();
      variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
      variancesLog->precision(10);


      nullspacesLog = new std::ofstream();
      nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
      nullspacesLog->precision(10);
    }
    else {
      nullspacesLog = 0;
      variancesLog = 0;
      DiagonalLog = 0;
      eigenALog = 0;
      eigenPLog = 0;
      eigenAllLog = 0;
      numsLog = 0;
      calibLog = 0;
    }

    assert(retstat != 293847);


    selectionMap = new float[wG[0] * hG[0]];

    coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
    coarseTracker = new CoarseTracker(wG[0], hG[0]);
    coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
    coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
    pixelSelector = new PixelSelector(wG[0], hG[0]);
    imuPropagation = new IMUPropagation();

    statistics_lastNumOptIts = 0;
    statistics_numDroppedPoints = 0;
    statistics_numActivatedPoints = 0;
    statistics_numCreatedPoints = 0;
    statistics_numForceDroppedResBwd = 0;
    statistics_numForceDroppedResFwd = 0;
    statistics_numMargResFwd = 0;
    statistics_numMargResBwd = 0;

    lastCoarseRMSE.setConstant(100);

    currentMinActDist = 2;
    initialized = false;


    ef = new EnergyFunctional();
    ef->red = &this->treadReduce;

    isLost = false;
    initFailed = false;


    needNewKFAfter = -1;

    linearizeOperation = true;
    runMapping = true;
    mappingThread = boost::thread(&FullSystem::mappingLoop, this);
    lastRefStopID = 0;


    minIdJetVisDebug = -1;
    maxIdJetVisDebug = -1;
    minIdJetVisTracker = -1;
    maxIdJetVisTracker = -1;

  }

  FullSystem::~FullSystem() {
    blockUntilMappingIsFinished();

    if (setting_logStuff) {
      calibLog->close();
      delete calibLog;
      numsLog->close();
      delete numsLog;
      coarseTrackingLog->close();
      delete coarseTrackingLog;
      //errorsLog->close(); delete errorsLog;
      eigenAllLog->close();
      delete eigenAllLog;
      eigenPLog->close();
      delete eigenPLog;
      eigenALog->close();
      delete eigenALog;
      DiagonalLog->close();
      delete DiagonalLog;
      variancesLog->close();
      delete variancesLog;
      nullspacesLog->close();
      delete nullspacesLog;
    }

    delete[] selectionMap;

    for (FrameShell *s : allFrameHistory)
      delete s;
    for (FrameShell *s : allFrameHistoryRight)
      delete s;
    for (FrameHessian *fh : unmappedTrackedFrames)
      delete fh;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    delete coarseInitializer;
    delete pixelSelector;
    delete ef;
    delete imuPropagation;
  }

  void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH) {

  }

  void FullSystem::setGammaFunction(float *BInv) {
    if (BInv == 0) return;

    // copy BInv.
    memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);


    // invert.
    for (int i = 1; i < 255; i++) {
      // find val, such that Binv[val] = i.
      // I dont care about speed for this, so do it the stupid way.

      for (int s = 1; s < 255; s++) {
        if (BInv[s] <= i && BInv[s + 1] >= i) {
          Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
          break;
        }
      }
    }
    Hcalib.B[0] = 0;
    Hcalib.B[255] = 255;
  }


  void FullSystem::printResult(std::string file) {
    boost::unique_lock<boost::mutex> lock(trackMutex);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    std::ofstream myfile;
    myfile.open(file.c_str());
    myfile << std::setprecision(15);

    for (FrameShell *s : allFrameHistory) {
      if (!s->poseValid) continue;

      if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

      myfile << s->timestamp <<
             " " << s->T_WC.translation().transpose() <<
             " " << s->T_WC.so3().unit_quaternion().x() <<
             " " << s->T_WC.so3().unit_quaternion().y() <<
             " " << s->T_WC.so3().unit_quaternion().z() <<
             " " << s->T_WC.so3().unit_quaternion().w() << "\n";
    }
    myfile.close();
  }

#if STEREO_MODE

  Vec4 FullSystem::trackNewCoarseStereo(FrameHessian *fh, FrameHessian *fhRight) {

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for (IOWrap::Output3DWrapper *ow : outputWrapper)
      ow->pushLiveFrame(fh);


    FrameHessian *lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0, 0);
    AffLight aff_last_2_l_r = AffLight(0, 0);

    std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
    if (allFrameHistory.size() == 2) {

//    if (false) {
      initializeFromInitializerStereo(fh);

      lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double, 3, 1>::Zero()));

      for (float rotDelta = 0.02; rotDelta < 0.1; rotDelta = rotDelta + 0.02) {
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(
            SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
      }

      coarseTracker->makeK(&Hcalib);
      coarseTracker->setCTRefForFirstFrame(frameHessians);
      lastF = coarseTracker->lastRef;
    }
    else {

      FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
      FrameShell *slastRight = allFrameHistoryRight[allFrameHistoryRight.size() - 2];
      FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
      SE3 slast_2_sprelast;
      SE3 lastF_2_slast;
      {  // lock on global pose consistency!
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        slast_2_sprelast = sprelast->T_WC.inverse() * slast->T_WC;
        lastF_2_slast = slast->T_WC.inverse() * lastF->shell->T_WC;
        aff_last_2_l = slast->aff_g2l;
        aff_last_2_l_r = slastRight->aff_g2l;
      }
      SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


      // get last delta-movement.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);  // assume constant motion.
      lastF_2_fh_tries.push_back(
          fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);  // assume double motion (frame skipped)
      lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
      lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
      lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


      // just try a TON of different initializations (all rotations). In the end,
      // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
      // also, if tracking rails here we loose, so we really, really want to avoid that.
      for (float rotDelta = 0.02; rotDelta < 0.1; rotDelta = rotDelta + 0.02) {
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
      }

      if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
        lastF_2_fh_tries.clear();
        lastF_2_fh_tries.push_back(SE3());
      }
    }


    Vec3 flowVecs = Vec3(100, 100, 100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0, 0);
    AffLight aff_g2l_r = AffLight(0, 0);


    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations = 0;
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
      AffLight aff_g2l_this = aff_last_2_l;
      AffLight aff_g2l_r_this = aff_last_2_l_r;
      SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
      bool trackingIsGood = coarseTracker->trackNewestCoarseStereo(
          fh, fhRight, lastF_2_fh_this, aff_g2l_this, aff_g2l_r_this,
          pyrLevelsUsed - 1,
          achievedRes);  // in each level has to be at least as good as the last try.
      tryIterations++;

      if (i != 0) {
        printf(
            "RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f), "
                "\naverage pixel photometric error for each level: %f %f %f %f %f -> %f %f %f %f %f \n",
            i,
            i, pyrLevelsUsed - 1,
            aff_g2l_this.a, aff_g2l_this.b,
            achievedRes[0],
            achievedRes[1],
            achievedRes[2],
            achievedRes[3],
            achievedRes[4],
            coarseTracker->lastResiduals[0],
            coarseTracker->lastResiduals[1],
            coarseTracker->lastResiduals[2],
            coarseTracker->lastResiduals[3],
            coarseTracker->lastResiduals[4]);
      }


      // do we have a new winner?
      if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
          !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
        //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
        flowVecs = coarseTracker->lastFlowIndicators;
        aff_g2l = aff_g2l_this;
        aff_g2l_r = aff_g2l_r_this;
        lastF_2_fh = lastF_2_fh_this;
        haveOneGood = true;
      }

      // take over achieved res (always).
      if (haveOneGood) {
        for (int i = 0; i < 5; i++) {
          if (!std::isfinite((float) achievedRes[i]) ||
              achievedRes[i] > coarseTracker->lastResiduals[i])  // take over if achievedRes is either bigger or NAN.
            achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }


      if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
        break;

    }

    if (!haveOneGood) {
      printf("BIG ERROR! tracking failed entirely. Take predicted pose and hope we may somehow recover.\n");
      flowVecs = Vec3(0, 0, 0);
      aff_g2l = aff_last_2_l;
      aff_g2l_r = aff_last_2_l_r;
      lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
    //- And also calculate right frame
    fhRight->shell->aff_g2l = aff_g2l_r;


    if (coarseTracker->firstCoarseRMSE < 0)
      coarseTracker->firstCoarseRMSE = achievedRes[0];

    if (!setting_debugout_runquiet)
      printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure,
             achievedRes[0]);


    if (setting_logStuff) {
      (*coarseTrackingLog) << std::setprecision(16)
                           << fh->shell->id << " "
                           << fh->shell->timestamp << " "
                           << fh->ab_exposure << " "
                           << fh->shell->T_WC.log().transpose() << " "
                           << aff_g2l.a << " "
                           << aff_g2l.b << " "
                           << achievedRes[0] << " "
                           << tryIterations << "\n";
    }


    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
  }

#else

  Vec4 FullSystem::trackNewCoarse(FrameHessian *fh) {

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for (IOWrap::Output3DWrapper *ow : outputWrapper)
      ow->pushLiveFrame(fh);


    FrameHessian *lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0, 0);

    std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
    if (allFrameHistory.size() == 2)
      for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) lastF_2_fh_tries.push_back(SE3());
    else {
      FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
      FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
      SE3 slast_2_sprelast;
      SE3 lastF_2_slast;
      {  // lock on global pose consistency!
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        slast_2_sprelast = sprelast->T_WC.inverse() * slast->T_WC;
        lastF_2_slast = slast->T_WC.inverse() * lastF->shell->T_WC;
        aff_last_2_l = slast->aff_g2l;
      }
      SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


      // get last delta-movement.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);  // assume constant motion.
      lastF_2_fh_tries.push_back(
          fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);  // assume double motion (frame skipped)
      lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
      lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
      lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


      // just try a TON of different initializations (all rotations). In the end,
      // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
      // also, if tracking rails here we loose, so we really, really want to avoid that.
      for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                                                              Vec3(0, 0,
                                                                                   0)));      // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                       Vec3(0, 0, 0)));  // assume constant motion.
      }

      if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
        lastF_2_fh_tries.clear();
        lastF_2_fh_tries.push_back(SE3());
      }
    }


    Vec3 flowVecs = Vec3(100, 100, 100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0, 0);


    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations = 0;
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
      AffLight aff_g2l_this = aff_last_2_l;
      SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
      bool trackingIsGood = coarseTracker->trackNewestCoarse(
          fh, lastF_2_fh_this, aff_g2l_this,
          pyrLevelsUsed - 1,
          achievedRes);  // in each level has to be at least as good as the last try.
      tryIterations++;

      if (i != 0) {
        printf(
            "RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
            i,
            i, pyrLevelsUsed - 1,
            aff_g2l_this.a, aff_g2l_this.b,
            achievedRes[0],
            achievedRes[1],
            achievedRes[2],
            achievedRes[3],
            achievedRes[4],
            coarseTracker->lastResiduals[0],
            coarseTracker->lastResiduals[1],
            coarseTracker->lastResiduals[2],
            coarseTracker->lastResiduals[3],
            coarseTracker->lastResiduals[4]);
      }


      // do we have a new winner?
      if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
          !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
        //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
        flowVecs = coarseTracker->lastFlowIndicators;
        aff_g2l = aff_g2l_this;
        lastF_2_fh = lastF_2_fh_this;
        haveOneGood = true;
      }

      // take over achieved res (always).
      if (haveOneGood) {
        for (int i = 0; i < 5; i++) {
          if (!std::isfinite((float) achievedRes[i]) ||
              achievedRes[i] > coarseTracker->lastResiduals[i])  // take over if achievedRes is either bigger or NAN.
            achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }


      if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
        break;

    }

    if (!haveOneGood) {
      printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
      flowVecs = Vec3(0, 0, 0);
      aff_g2l = aff_last_2_l;
      lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;


    if (coarseTracker->firstCoarseRMSE < 0)
      coarseTracker->firstCoarseRMSE = achievedRes[0];

    if (!setting_debugout_runquiet)
      printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure,
             achievedRes[0]);


    if (setting_logStuff) {
      (*coarseTrackingLog) << std::setprecision(16)
                           << fh->shell->id << " "
                           << fh->shell->timestamp << " "
                           << fh->ab_exposure << " "
                           << fh->shell->T_WC.log().transpose() << " "
                           << aff_g2l.a << " "
                           << aff_g2l.b << " "
                           << achievedRes[0] << " "
                           << tryIterations << "\n";
    }


    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
  }
#endif

  void FullSystem::traceNewCoarseNonKey(FrameHessian *fh, FrameHessian *fhRight) {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // new idepth after refinement
    float idepth_min_update = 0;
    float idepth_max_update = 0;

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

    Mat33f Ki = K.inverse();

    for (FrameHessian *host : frameHessians)    // go through all active frames
    {
      int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

      SE3 hostToNew = fh->PRE_T_CW * host->PRE_T_WC;
      Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
      Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
      Vec3f Kt = K * hostToNew.translation().cast<float>();
      Vec3f t = hostToNew.translation().cast<float>();

      Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(),
                                              fh->aff_g2l()).cast<float>();

      for (ImmaturePoint *ph : host->immaturePoints) {
        //- Do temporal stereo match
        ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

        if (phTrackStatus == ImmaturePointStatus::IPS_GOOD) {
          ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, ph->my_type, &Hcalib);

          Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
          float idepth_min_project = 1.0f / ptpMin[2];
          Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
          float idepth_max_project = 1.0f / ptpMax[2];

          phNonKey->idepth_min = idepth_min_project;
          phNonKey->idepth_max = idepth_max_project;
          phNonKey->u_stereo = phNonKey->u;
          phNonKey->v_stereo = phNonKey->v;
          phNonKey->idepth_min_stereo = phNonKey->idepth_min;
          phNonKey->idepth_max_stereo = phNonKey->idepth_max;

          //- Do static stereo match from left to right
          ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fhRight, K, true);

          if (phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD) {
            ImmaturePoint *phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1),
                                                             fhRight, ph->my_type, &Hcalib);

            phNonKeyRight->u_stereo = phNonKeyRight->u;
            phNonKeyRight->v_stereo = phNonKeyRight->v;
            phNonKeyRight->idepth_min_stereo = 0;
            phNonKeyRight->idepth_max_stereo = NAN;

            ImmaturePointStatus phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, K, false);

            float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
            float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

            if (u_stereo_delta > 1 && disparity < 10) {
              ph->lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }
            else {
              Vec3f pinverse_min =
                  KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
              idepth_min_update = 1.0f / pinverse_min(2);

              Vec3f pinverse_max =
                  KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
              idepth_max_update = 1.0f / pinverse_max(2);

              ph->idepth_min = idepth_min_update;
              ph->idepth_max = idepth_max_update;

            }
            delete phNonKeyRight;
          }
          delete phNonKey;
        }

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
        trace_total++;
      }
    }
  }

  void FullSystem::traceNewCoarseKey(FrameHessian *fh) {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

    for (FrameHessian *host : frameHessians)    // go through all active frames
    {

      SE3 hostToNew = fh->PRE_T_CW * host->PRE_T_WC;
      Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
      Vec3f Kt = K * hostToNew.translation().cast<float>();

      Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(),
                                              fh->aff_g2l()).cast<float>();

      for (ImmaturePoint *ph : host->immaturePoints) {
        ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
        trace_total++;
      }
    }
  }


  void FullSystem::activatePointsMT_Reductor(
      std::vector<PointHessian *> *optimized,
      std::vector<ImmaturePoint *> *toOptimize,
      int min, int max, Vec10 *stats, int tid) {
    ImmaturePointTemporaryResidual *tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    for (int k = min; k < max; k++) {
      (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
    }
    delete[] tr;
  }


  void FullSystem::activatePointsMT() {

    if (ef->nPoints < setting_desiredPointDensity * 0.66)
      currentMinActDist -= 0.8;
    if (ef->nPoints < setting_desiredPointDensity * 0.8)
      currentMinActDist -= 0.5;
    else if (ef->nPoints < setting_desiredPointDensity * 0.9)
      currentMinActDist -= 0.2;
    else if (ef->nPoints < setting_desiredPointDensity)
      currentMinActDist -= 0.1;

    if (ef->nPoints > setting_desiredPointDensity * 1.5)
      currentMinActDist += 0.8;
    if (ef->nPoints > setting_desiredPointDensity * 1.3)
      currentMinActDist += 0.5;
    if (ef->nPoints > setting_desiredPointDensity * 1.15)
      currentMinActDist += 0.2;
    if (ef->nPoints > setting_desiredPointDensity)
      currentMinActDist += 0.1;

    if (currentMinActDist < 0) currentMinActDist = 0;
    if (currentMinActDist > 4) currentMinActDist = 4;

    if (!setting_debugout_runquiet)
      printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
             currentMinActDist, (int) (setting_desiredPointDensity), ef->nPoints);

    FrameHessian *newestHs = frameHessians.back();

    // make dist map.
    coarseDistanceMap->makeK(&Hcalib);
    coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

    //coarseTracker->debugPlotDistMap("distMap");

    std::vector<ImmaturePoint *> toOptimize;
    toOptimize.reserve(20000);

    for (FrameHessian *host : frameHessians)    // go through all active frames
    {
      if (host == newestHs) continue;

      SE3 fhToNew = newestHs->PRE_T_CW * host->PRE_T_WC;
      Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
      Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


      for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
        ImmaturePoint *ph = host->immaturePoints[i];
        ph->idxInImmaturePoints = i;

        // delete points that have never been traced successfully, or that are outlier on the last trace.
        if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
//				immature_invalid_deleted++;
          // remove point.
          delete ph;
          host->immaturePoints[i] = 0;
          continue;
        }

        // can activate only if this is true.
        bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                            || ph->lastTraceStatus == IPS_SKIPPED
                            || ph->lastTraceStatus == IPS_BADCONDITION
                            || ph->lastTraceStatus == IPS_OOB)
                           && ph->lastTracePixelInterval < 8
                           && ph->quality > setting_minTraceQuality
                           && (ph->idepth_max + ph->idepth_min) > 0;


        // if I cannot activate the point, skip it. Maybe also delete it.
        if (!canActivate) {
          // if point will be out afterwards, delete it instead.
          if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
//					immature_notReady_deleted++;
            delete ph;
            host->immaturePoints[i] = 0;
          }
//				immature_notReady_skipped++;
          continue;
        }


        // see if we need to activate point due to distance map.
        Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
        int u = ptp[0] / ptp[2] + 0.5f;
        int v = ptp[1] / ptp[2] + 0.5f;

        if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {

          float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] + (ptp[0] - floorf((float) (ptp[0])));

          if (dist >= currentMinActDist * ph->my_type) {
            coarseDistanceMap->addIntoDistFinal(u, v);
            toOptimize.push_back(ph);
          }
        }
        else {
          delete ph;
          host->immaturePoints[i] = 0;
        }
      }
    }

    std::vector<PointHessian *> optimized;
    optimized.resize(toOptimize.size());

    if (multiThreading)
      treadReduce.reduce(
          boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0,
          toOptimize.size(), 50);

    else
      activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

    for (unsigned k = 0; k < toOptimize.size(); k++) {
      PointHessian *newpoint = optimized[k];
      ImmaturePoint *ph = toOptimize[k];

      if (newpoint != 0 && newpoint != (PointHessian *) ((long) (-1))) {
        newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
        newpoint->host->pointHessians.push_back(newpoint);
        ef->insertPoint(newpoint);
        for (PointFrameResidual *r : newpoint->residuals) {
          if (r->staticStereo) { //- static stereo residual
            ef->insertStaticResidual(r);
          }
          else
            ef->insertResidual(r);
        }
        assert(newpoint->efPoint != 0);
        delete ph;
      }
      else if (newpoint == (PointHessian *) ((long) (-1)) || ph->lastTraceStatus == IPS_OOB) {
        ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
        delete ph;
      }
      else {
        assert(newpoint == 0 || newpoint == (PointHessian *) ((long) (-1)));
      }
    }


    for (FrameHessian *host : frameHessians) {
      for (int i = 0; i < (int) host->immaturePoints.size(); i++) {
        if (host->immaturePoints[i] == 0) {
          host->immaturePoints[i] = host->immaturePoints.back();
          host->immaturePoints.pop_back();
          i--;
        }
      }
    }


  }

/*
  void FullSystem::activatePointsRight(FrameHessian *fh, FrameHessian *fhRight) {

    // From Left to Right

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();


    int counter = 0;

    for (ImmaturePoint *ipRight : fhRight->immaturePoints) {
      ipRight->u_stereo = ipRight->u;
      ipRight->v_stereo = ipRight->v;
      ipRight->idepth_min_stereo = ipRight->idepth_min = 0;
      ipRight->idepth_max_stereo = ipRight->idepth_max = NAN;

      //- From Right to Left
      ImmaturePointStatus phTraceLeftStatus = ipRight->traceStereo(fh, K, 0);

      if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD) {
        ImmaturePoint *ipLeft = new ImmaturePoint(ipRight->lastTraceUV(0), ipRight->lastTraceUV(1), fh,
                                                  ipRight->my_type, &Hcalib);

        ipLeft->u_stereo = ipLeft->u;
        ipLeft->v_stereo = ipLeft->v;
        ipLeft->idepth_min_stereo = ipRight->idepth_min = 0;
        ipLeft->idepth_max_stereo = ipRight->idepth_max = NAN;
//        std::cout << "idx: " << ipRight->idxInImmaturePoints << "\t Left." << std::endl;
        ImmaturePointStatus phTraceLeftStatus = ipLeft->traceStereo(fh, K, 0);

        float u_stereo_delta = abs(ipRight->u_stereo - ipLeft->lastTraceUV(0));
        float depth = 1.0f / ipRight->idepth_stereo;

        if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 &&
            depth < 70)    //original u_stereo_delta 1 depth < 70
        {

          ipRight->idepth_min = ipRight->idepth_min_stereo;
          ipRight->idepth_max = ipRight->idepth_max_stereo;

          //- convert immature point to point hessian
          ImmaturePointTemporaryResidual *r = new ImmaturePointTemporaryResidual();
          PointHessian *newph = optimizeImmaturePointRight(ipRight, r);

          if (newph != 0 && newph != (PointHessian *) ((long) (-1))) {
            newph->host->immaturePoints[ipRight->idxInImmaturePoints] = 0;
            newph->host->pointHessians.push_back(newph);
            ef->insertPoint(newph);
            for (PointFrameResidual *r : newph->residuals)
              ef->insertResidual(r);
            assert(newph->efPoint != 0);
            delete ipRight;
          } else if (newph == (PointHessian *) ((long) (-1)) || ipRight->lastTraceStatus == IPS_OOB) {
            delete ipRight;
            ipRight->host->immaturePoints[ipRight->idxInImmaturePoints] = 0;
          } else {
            assert(newph == 0 || newph == (PointHessian *) ((long) (-1)));
          }

          delete r;
          counter++;
        }
      }
    }

    //- Delete immature points
    for (int i = 0; i < (int) fhRight->immaturePoints.size(); i++) {
      if (fhRight->immaturePoints[i] == 0) {
        fhRight->immaturePoints[i] = fhRight->immaturePoints.back();
        fhRight->immaturePoints.pop_back();
        i--;
      }
    }

    //- Formally transform immaturePoints to pointHesssians
//    for (unsigned k = 0; k < toOptimize.size(); k++) {
//      PointHessian *newpoint = optimized[k];
//      ImmaturePoint *ph = toOptimize[k];
//
//      if (newpoint != 0 && newpoint != (PointHessian *) ((long) (-1))) {
//        newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
//        newpoint->host->pointHessians.push_back(newpoint);
//        ef->insertPoint(newpoint);
//        for (PointFrameResidual *r : newpoint->residuals)
//          ef->insertResidual(r);
//        assert(newpoint->efPoint != 0);
//        delete ph;
//      } else if (newpoint == (PointHessian *) ((long) (-1)) || ph->lastTraceStatus == IPS_OOB) {
//        delete ph;
//        ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
//      } else {
//        assert(newpoint == 0 || newpoint == (PointHessian *) ((long) (-1)));
//      }
//    }
//
//
//    for (FrameHessian *host : frameHessians) {
//      for (int i = 0; i < (int) host->immaturePoints.size(); i++) {
//        if (host->immaturePoints[i] == 0) {
//          host->immaturePoints[i] = host->immaturePoints.back();
//          host->immaturePoints.pop_back();
//          i--;
//        }
//      }
//    }


    return;
  }
*/

  void FullSystem::activatePointsOldFirst() {
    assert(false);
  }

  void FullSystem::flagPointsForRemoval() {
    assert(EFIndicesValid);

    std::vector<FrameHessian *> fhsToKeepPoints;
    std::vector<FrameHessian *> fhsToMargPoints;

    //if(setting_margPointVisWindow>0)
    {
      for (int i = ((int) frameHessians.size()) - 1; i >= 0 && i >= ((int) frameHessians.size()); i--)
        if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

      for (int i = 0; i < (int) frameHessians.size(); i++)
        if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
    }



    //ef->setAdjointsF();
    //ef->setDeltaF(&Hcalib);
    int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

    for (FrameHessian *host : frameHessians)    // go through all active frames
    {
      for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
        PointHessian *ph = host->pointHessians[i];
        if (ph == 0) continue;

        if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
          host->pointHessians[i] = 0;
          flag_nores++;
        }
        else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
          flag_oob++;
          if (ph->isInlierNew()) {
            flag_in++;
            int ngoodRes = 0;
            for (PointFrameResidual *r : ph->residuals) {
              r->resetOOB();
              if (r->staticStereo)
                r->linearizeStatic(&Hcalib);
              else
                r->linearize(&Hcalib);
              r->efResidual->isLinearized = false;
              r->applyRes(true);
              if (r->efResidual->isActive()) {
                r->efResidual->fixLinearizationF(ef);
                ngoodRes++;
              }
            }
            if (ph->idepth_hessian > setting_minIdepthH_marg) {
              flag_inin++;
              ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
              host->pointHessiansMarginalized.push_back(ph);
            }
            else {
              ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
              host->pointHessiansOut.push_back(ph);
            }


          }
          else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
            //printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
          }

          host->pointHessians[i] = 0;
        }
      }


      for (int i = 0; i < (int) host->pointHessians.size(); i++) {
        if (host->pointHessians[i] == 0) {
          host->pointHessians[i] = host->pointHessians.back();
          host->pointHessians.pop_back();
          i--;
        }
      }
    }

  }

  void FullSystem::stereoMatch(ImageAndExposure *image, ImageAndExposure *imageRight, int id, cv::Mat &idepthMap) {
    // =========================== add into allFrameHistory =========================
    FrameHessian *fh = new FrameHessian();
    FrameHessian *fhRight = new FrameHessian();
    FrameShell *shell = new FrameShell();
    shell->T_WC = SE3();    // no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0, 0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id; // id passed into DSO
    fh->shell = shell;
    fhRight->shell = shell;

    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);
    fhRight->ab_exposure = imageRight->exposure_time;
    fhRight->makeImages(imageRight->image, &Hcalib);

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();


    int counter = 0;

    makeNewTraces(fh, 0);

    unsigned char *idepthMapPtr = idepthMap.data;

    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;
    std::vector<cv::DMatch> matches;

    for (ImmaturePoint *ph : fh->immaturePoints) {
      ph->u_stereo = ph->u;
      ph->v_stereo = ph->v;
      ph->idepth_min_stereo = ph->idepth_min = 0;
      ph->idepth_max_stereo = ph->idepth_max = NAN;


//      std::cout << "idx: " << ph->idxInImmaturePoints << "\t Right." << std::endl;
      ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fhRight, K, 1);
      /*
       *   enum ImmaturePointStatus {
            IPS_GOOD = 0,          // traced well and good
            IPS_OOB,          // OOB: end tracking & marginalize!
            IPS_OUTLIER,        // energy too high: if happens again: outlier!
            IPS_SKIPPED,        // traced well and good (but not actually traced).
            IPS_BADCONDITION,      // not traced because of bad condition.
            IPS_UNINITIALIZED
           };      // not even traced once.
       */
//      std::cout << "phTraceRightStatus: " << phTraceRightStatus << std::endl;
      if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD) {
        ImmaturePoint *phRight = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fhRight, ph->my_type,
                                                   &Hcalib);

        phRight->u_stereo = phRight->u;
        phRight->v_stereo = phRight->v;
        phRight->idepth_min_stereo = ph->idepth_min = 0;
        phRight->idepth_max_stereo = ph->idepth_max = NAN;
//        std::cout << "idx: " << ph->idxInImmaturePoints << "\t Left." << std::endl;
        ImmaturePointStatus phTraceLeftStatus = phRight->traceStereo(fh, K, 0);

        float u_stereo_delta = abs(ph->u_stereo - phRight->lastTraceUV(0));
        float depth = 1.0f / ph->idepth_stereo;

        if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 &&
            depth < setting_acceptStaticDepthFactor * baseline)    //original u_stereo_delta 1 depth < 70
        {
//          if (ph->u - ph->lastTraceUV(0) > 0) {
          keypoints_left.emplace_back(ph->u, ph->v, 1);
          keypoints_right.emplace_back(ph->lastTraceUV(0), ph->lastTraceUV(1), 1);
          matches.emplace_back(keypoints_left.size() - 1, keypoints_right.size() - 1, 1.0f);
//          }

          ph->idepth_min = ph->idepth_min_stereo;
          ph->idepth_max = ph->idepth_max_stereo;

          *((float *) (idepthMapPtr + int(ph->v) * idepthMap.step) + (int) ph->u * 3) = ph->idepth_stereo;
          *((float *) (idepthMapPtr + int(ph->v) * idepthMap.step) + (int) ph->u * 3 + 1) = ph->idepth_min;
          *((float *) (idepthMapPtr + int(ph->v) * idepthMap.step) + (int) ph->u * 3 + 2) = ph->idepth_max;

          counter++;
        }
      }
    }

//    std::sort(error.begin(), error.end());
//    std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
//              0.5 <<" "<<error[error.size()*0.5].first<<" "<<
//              0.75 <<" "<<error[error.size()*0.75].first<<" "<<
//              0.1 <<" "<<error.back().first << std::endl;

//    for(int i = 0; i < error.size(); i++)
//        std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

    std::cout << " frameID " << id << " got good matches " << counter << std::endl;

    cv::Mat matLeft(image->h, image->w, CV_32F, image->image);
    cv::Mat matRight(imageRight->h, imageRight->w, CV_32F, imageRight->image);
    matLeft.convertTo(matLeft, CV_8UC3);
    matRight.convertTo(matRight, CV_8UC3);

    float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

    cv::Mat matMatches;
    if (false) {
      cv::drawMatches(matLeft, keypoints_left, matRight, keypoints_right, matches, matMatches);
    }
    else {
      matMatches.create(cv::Size(matLeft.cols + matRight.cols, matLeft.rows), CV_MAKETYPE(matLeft.depth(), 3));
      cv::Mat outLeft = matMatches(cv::Rect(0, 0, matLeft.cols, matLeft.rows));
      cv::Mat outRight = matMatches(cv::Rect(matLeft.cols, 0, matRight.cols, matRight.rows));
      cv::RNG rng(0);
      cv::cvtColor(matLeft, outLeft, cv::COLOR_GRAY2BGR);
      cv::cvtColor(matRight, outRight, cv::COLOR_GRAY2BGR);
      for (int i = 0; i < matches.size(); ++i) {
        keypoints_right[i].pt.x += image->w;
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawMarker(matMatches, keypoints_left[i].pt, color, cv::MARKER_SQUARE, 3);
        cv::drawMarker(matMatches, keypoints_right[i].pt, color, cv::MARKER_SQUARE, 3);

        //- 1. Draw infinite depth point in right image.
//        keypoints_left[i].pt.x += image->w;
//        cv::drawMarker(matMatches, keypoints_left[i].pt, color, cv::MARKER_SQUARE, 3);
        //- 2. Draw search line in right image.
        keypoints_right[i].pt = keypoints_left[i].pt;
        float min_v = keypoints_right[i].pt.x - maxPixSearch;
        min_v = min_v > 0 ? min_v : 0;
        keypoints_right[i].pt.x = min_v + image->w;
        keypoints_left[i].pt.x += image->w;
        cv::line(matMatches, keypoints_right[i].pt, keypoints_left[i].pt, color);
        //- 3. Draw search line in left image.
//        keypoints_right[i].pt.x -= image->w;
//        cv::line(matMatches, keypoints_right[i].pt, keypoints_left[i].pt, color);

      }
    }

    cv::imshow("matches", matMatches);
    cv::imwrite("matches.png", matMatches);
    cv::waitKey(0);

    delete fh;
    delete fhRight;

    return;
  }


  void FullSystem::stereoMatch(FrameHessian *fh, FrameHessian *fhRight) {

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

//    std::vector<cv::KeyPoint> keypoints_left;
//    std::vector<cv::KeyPoint> keypoints_right;
//    std::vector<cv::DMatch> matches;
//    float debugKeepPercentage = 0.2;

    for (ImmaturePoint *ip : fh->immaturePoints) {
      ip->u_stereo = ip->u;
      ip->v_stereo = ip->v;
      ip->idepth_min_stereo = ip->idepth_min = 0;
      ip->idepth_max_stereo = ip->idepth_max = NAN;


//      std::cout << "idx: " << ip->idxInImmaturePoints << "\t Right." << std::endl;
      ImmaturePointStatus phTraceRightStatus = ip->traceStereo(fhRight, K, 1);

      if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD) {
        ImmaturePoint *ipRight = new ImmaturePoint(ip->lastTraceUV(0), ip->lastTraceUV(1), fhRight, ip->my_type,
                                                   &Hcalib);

        ipRight->u_stereo = ipRight->u;
        ipRight->v_stereo = ipRight->v;
        ipRight->idepth_min_stereo = ip->idepth_min = 0;
        ipRight->idepth_max_stereo = ip->idepth_max = NAN;
//        std::cout << "idx: " << ip->idxInImmaturePoints << "\t Left." << std::endl;
        ImmaturePointStatus phTraceLeftStatus = ipRight->traceStereo(fh, K, 0);

        float u_stereo_delta = abs(ip->u_stereo - ipRight->lastTraceUV(0));
        float depth = 1.0f / ip->idepth_stereo;

        if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 &&
            depth < setting_acceptStaticDepthFactor * baseline && depth > 20 * baseline) //original u_stereo_delta 1 depth < 70
        {

          ip->idepth_min = ip->idepth_min_stereo;
          ip->idepth_max = ip->idepth_max_stereo;

//          if (rand() / (float) RAND_MAX > debugKeepPercentage) continue;
//          keypoints_left.emplace_back(ip->u, ip->v, 1);
//          keypoints_right.emplace_back(ip->lastTraceUV(0), ip->lastTraceUV(1), 1);
//          matches.emplace_back(keypoints_left.size() - 1, keypoints_right.size() - 1, 1.0f);
        }
      }
    }

//    cv::Mat matLeft(hG[0], wG[0], CV_32FC3, fh->dI);
//    cv::Mat matRight(hG[0], wG[0], CV_32FC3, fhRight->dI);
//    matLeft.convertTo(matLeft, CV_8UC3);
//    matRight.convertTo(matRight, CV_8UC3);
//
//    cv::Mat matMatches;
//    cv::drawMatches(matLeft, keypoints_left, matRight, keypoints_right, matches, matMatches);
//    cv::imshow("matches", matMatches);
//    cv::waitKey(0);
  }

  void FullSystem::addActiveFrame(ImageAndExposure *image, ImageAndExposure *imageRight,
                                  std::vector<IMUMeasurement> &imuMeasurements, int id) {


//    cv::Mat matLeft(image->h, image->w, CV_32F, image->image);
//    cv::Mat matRight(imageRight->h, imageRight->w, CV_32F, imageRight->image);
//
//
//    matLeft.convertTo(matLeft, CV_8UC3);
//    matRight.convertTo(matRight, CV_8UC3);
//
//    cv::Mat matShow(image->h, image->w * 2, CV_8UC3);
//
//    matShow.rowRange(0, image->h).colRange(0, image->w) = matLeft;
//    matShow.rowRange(0, image->h).colRange(image->w, image->w * 2) = matRight;
//
//
//    cv::imshow("LR", matShow);
//    cv::waitKey(0);

    if (isLost) return;
    boost::unique_lock<boost::mutex> lock(trackMutex);

//    printf("addActive Frame left timestamp: %lf, right timestamp: %lf\n", image->timestamp, imageRight->timestamp);

    //- Checkout if camera intrinsics changed.
//    printf("Hcalib: %f, %f, %f, %f\n", Hcalib.fxl(), Hcalib.fyl(), Hcalib.cxl(), Hcalib.cyl());

    // =========================== add into allFrameHistory =========================
    FrameHessian *fh = new FrameHessian();
    FrameShell *shell = new FrameShell();
    shell->T_WC = SE3();    // no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0, 0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
    fh->shell = shell;
    allFrameHistory.push_back(shell);
    //- Right Image
    FrameHessian *fhRight = new FrameHessian();
    FrameShell *shellRight = new FrameShell();
    shellRight->aff_g2l = AffLight(0, 0); //- only use the afflight parametere of right shell
    fhRight->shell = shellRight;
    allFrameHistoryRight.push_back(shellRight);

    fh->rightFrame = fhRight;
    fhRight->leftFrame = fh;

//    printf("addActiveFrame allFrameHistory.size(): %ld\n", allFrameHistory.size());


    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);
    fhRight->ab_exposure = imageRight->exposure_time;
    fhRight->makeImages(imageRight->image, &Hcalib);
    //- FrameHessian::makeImages() just calculate some image gradient.

//    printf("frameHessians.size(), frameHessiansRight.size(): %ld, %ld\n",
//           frameHessians.size(), frameHessiansRight.size());

    if (!initialized) {
#if STEREO_MODE
      // use initializer!
      if (coarseInitializer->frameID < 0)  // first frame set. fh is kept by coarseInitializer.
      {
        //- Initialize IMU
        Sophus::Quaterniond q_WS = imuPropagation->initializeRollPitchFromMeasurements(imuMeasurements);
        q_WS.setIdentity();
        // T_WS * T_SC0 = T_WC0
        coarseInitializer->T_WC_ini = SE3(Sophus::Quaterniond::Identity(), Vec3(0, 0, 0)) * T_SC0;
        //- Add the First frame to the corseInitializer.
        coarseInitializer->setFirstStereo(&Hcalib, fh, fhRight);

        initialized = true;

        fh->shell->aff_g2l = AffLight(0, 0);
        fh->rightFrame->shell->aff_g2l = AffLight(0, 0);
        fh->shell->T_WC = coarseInitializer->T_WC_ini;
        fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l, fh->rightFrame->shell->aff_g2l);

        fhRight->shell->aff_g2l = fhRight->shell->aff_g2l;
        fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
//        fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
      }
//      else if (coarseInitializer->trackFrame(fh, outputWrapper))  // if SNAPPED
//      {
//        initializeFromInitializer(fh);
//        lock.unlock();
//        deliverTrackedFrame(fh, fhRight, true);
//      }
//      else {
//        // if still initializing
//        fh->shell->poseValid = false;
//        delete fh;
//      }
      return;
#else
      // use initializer!
      if (coarseInitializer->frameID < 0)  // first frame set. fh is kept by coarseInitializer.
      {
        //- Initialize IMU
        Sophus::Quaterniond q_WS = imuPropagation->initializeRollPitchFromMeasurements(imuMeasurements);
        q_WS.setIdentity();
        // T_WS * T_SC0 = T_WC0
//        coarseInitializer->T_WC_ini = SE3(Sophus::Quaterniond::Identity(), Vec3(0, 0, 0)) * T_SC0;
        coarseInitializer->T_WC_ini = SE3();
        //- Add the First frame to the corseInitializer.
        coarseInitializer->setFirst(&Hcalib, fh);
      }
      else if (coarseInitializer->trackFrame(fh, outputWrapper))  // if SNAPPED
      {
        initializeFromInitializer(fh);
        lock.unlock();
        deliverTrackedFrame(fh, fhRight, true);
      }
      else {
        // if still initializing
        fh->shell->poseValid = false;
        delete fh;
      }
      return;
#endif
    }
    else  // do front-end operation.
    {
      // =========================== SWAP tracking reference?. =========================
      if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        CoarseTracker *tmp = coarseTracker;
        coarseTracker = coarseTracker_forNewKF;
        coarseTracker_forNewKF = tmp;
      }

#if STEREO_MODE
      Vec4 tres = trackNewCoarseStereo(fh, fhRight);
#else
      Vec4 tres = trackNewCoarse(fh);
#endif
      if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) || !std::isfinite((double) tres[2]) ||
          !std::isfinite((double) tres[3])) {
        printf("Initial Tracking failed: LOST!\n");
        isLost = true;
        delete fh;
        delete fhRight;
        return;
      }

      bool needToMakeKF = false;
      if (setting_keyframesPerSecond > 0) {
        needToMakeKF = allFrameHistory.size() == 1 ||
                       (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
                       0.95f / setting_keyframesPerSecond;
      }
      else {
        Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                   coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

        // BRIGHTNESS CHECK
        needToMakeKF = allFrameHistory.size() == 1 ||
                       setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double) tres[1]) / (wG[0] + hG[0]) +
                       setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double) tres[2]) / (wG[0] + hG[0]) +
                       setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double) tres[3]) / (wG[0] + hG[0]) +
                       setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float) refToFh[0])) > 1 ||
                       2 * coarseTracker->firstCoarseRMSE < tres[0];

      }


      for (IOWrap::Output3DWrapper *ow : outputWrapper)
        ow->publishCamPose(fh->shell, &Hcalib);


      lock.unlock();
      deliverTrackedFrame(fh, fhRight, needToMakeKF);
      return;
    }
  }

  void FullSystem::deliverTrackedFrame(FrameHessian *fh, FrameHessian *fhRight, bool needKF) {


    if (linearizeOperation) {
      if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
        printf("allFrameHistory.size() == %d\n", (int) allFrameHistory.size());
        MinimalImageF3 img(wG[0], hG[0], fh->dI);
        IOWrap::displayImage("frameToTrack", &img);
        while (true) {
          char k = IOWrap::waitKey(0);
          if (k == ' ') break;
          handleKey(k);
        }
        lastRefStopID = coarseTracker->refFrameID;
      }
      else handleKey(IOWrap::waitKey(1));


      if (needKF) makeKeyFrame(fh, fhRight);
      else makeNonKeyFrame(fh, fhRight);
    }
    else {
      boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
      unmappedTrackedFrames.push_back(fh);
      unmappedTrackedFramesRight.push_back(fhRight);
      if (needKF) needNewKFAfter = fh->shell->trackingRef->id;
      trackedFrameSignal.notify_all();

      while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
        mappedFrameSignal.wait(lock);
      }

      lock.unlock();
    }
  }

  void FullSystem::mappingLoop() {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    while (runMapping) {
      while (unmappedTrackedFrames.size() == 0) {
        trackedFrameSignal.wait(lock);
        if (!runMapping) return;
      }

      FrameHessian *fh = unmappedTrackedFrames.front();
      unmappedTrackedFrames.pop_front();

      FrameHessian *fhRight = unmappedTrackedFramesRight.front();
      unmappedTrackedFramesRight.pop_front();


      // guaranteed to make a KF for the very first two tracked frames.
      if (allKeyFramesHistory.size() <= 2) {
        lock.unlock();
        makeKeyFrame(fh, fhRight);
        lock.lock();
        mappedFrameSignal.notify_all();
        continue;
      }

      if (unmappedTrackedFrames.size() > 3)
        needToKetchupMapping = true;


      if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
      {
        lock.unlock();
        makeNonKeyFrame(fh, fhRight);
        lock.lock();

        if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
          FrameHessian *fh = unmappedTrackedFrames.front();
          unmappedTrackedFrames.pop_front();
          FrameHessian *fhRight = unmappedTrackedFramesRight.front();
          unmappedTrackedFramesRight.pop_front();
#if STEREO_MODE
          {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l, fh->rightFrame->shell->aff_g2l);

            fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
//            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
          }
#else
          {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l);

            fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
          }
#endif
          delete fh;
        }

      }
      else {
        if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) {
          lock.unlock();
          makeKeyFrame(fh, fhRight);
          needToKetchupMapping = false;
          lock.lock();
        }
        else {
          lock.unlock();
          makeNonKeyFrame(fh, fhRight);
          lock.lock();
        }
      }
      mappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
  }

  void FullSystem::blockUntilMappingIsFinished() {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    runMapping = false;
    trackedFrameSignal.notify_all();
    lock.unlock();

    mappingThread.join();

  }

  void FullSystem::makeNonKeyFrame(FrameHessian *fh, FrameHessian *fhRight) {
    // needs to be set by mapping thread. no lock required since we are in mapping thread.
#if STEREO_MODE
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      assert(fh->shell->trackingRef != 0);
      fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
      fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l, fh->rightFrame->shell->aff_g2l);

      fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
//            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
    }
#else
    {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l);

            fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
          }
#endif

//    traceNewCoarseKey(fh);
#if STEREO_MODE
    traceNewCoarseNonKey(fh, fhRight);
#else
    traceNewCoarseKey(fh);
#endif
    delete fh;
    delete fhRight;
  }

  void FullSystem::makeKeyFrame(FrameHessian *fh, FrameHessian *fhRight) {
    // needs to be set by mapping thread
#if STEREO_MODE
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      assert(fh->shell->trackingRef != 0);
      fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
      fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l, fh->rightFrame->shell->aff_g2l);

//      fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
//            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
    }
#else
    {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->T_WC = fh->shell->trackingRef->T_WC * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->T_WC.inverse(), fh->shell->aff_g2l);

            fhRight->shell->T_WC = fh->shell->T_WC * leftToRight_SE3.inverse();
            fhRight->setEvalPT_scaled(fhRight->shell->T_WC.inverse(), fhRight->shell->aff_g2l);
    }
#endif

    traceNewCoarseKey(fh);
//    traceNewCoarseNonKey(fh, fhRight);

    boost::unique_lock<boost::mutex> lock(mapMutex);

    // =========================== Flag Frames to be Marginalized. =========================
    flagFramesForMarginalization(fh);


    // =========================== add New Frame to Hessian Struct. =========================
    fh->idx = frameHessians.size();
    frameHessians.push_back(fh);
    frameHessiansRight.push_back(fhRight);
    fh->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(fh->shell);
    ef->insertFrame(fh, &Hcalib);

    setPrecalcValues();



    // =========================== add new residuals for old points =========================
//    printf("ef->nResiduals: %d\n", ef->nResiduals);
    int numFwdResAdded = 0;
    for (FrameHessian *fh1 : frameHessians)    // go through all active frames
    {
      if (fh1 == fh) continue;
      for (PointHessian *ph : fh1->pointHessians) {
        PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
        r->setState(ResState::IN);
        ph->residuals.push_back(r);
        ef->insertResidual(r);
        ph->lastResiduals[1] = ph->lastResiduals[0];
        ph->lastResiduals[0] = std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
        numFwdResAdded += 1;
      }
    }




    // =========================== Activate Points (& flag for marginalization). =========================
    activatePointsMT();
    ef->makeIDX();




    // =========================== OPTIMIZE ALL =========================

    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
    float rmse = optimize(setting_maxOptIterations); // setting_maxOptIterations == 6





    // =========================== Figure Out if INITIALIZATION FAILED =========================
    if (allKeyFramesHistory.size() <= 4) {
      if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
        printf("I THINK INITIALIZATION FAILED! Resetting.\n");
        initFailed = true;
      }
      if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
        printf("I THINK INITIALIZATION FAILED! Resetting.\n");
        initFailed = true;
      }
      if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
        printf("I THINK INITIALIZATION FAILED! Resetting.\n");
        initFailed = true;
      }
    }


    if (isLost) return;




    // =========================== REMOVE OUTLIER =========================
    removeOutliers();


    {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
      coarseTracker_forNewKF->makeK(&Hcalib);
      coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);


      coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
      coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
    }


    debugPlot("post Optimize");

    // =========================== (Activate-)Marginalize Points =========================
    flagPointsForRemoval();
    ef->dropPointsF();
    getNullspaces(
        ef->lastNullspaces_pose,
        ef->lastNullspaces_scale,
        ef->lastNullspaces_affA,
        ef->lastNullspaces_affB);
    ef->marginalizePointsF();



    // =========================== add new Immature points & new residuals =========================
    makeNewTraces(fh, 0);
#if STEREO_MODE
    //- use right frame to initialize the depth of fh->immaturePoints
    stereoMatch(fh, fhRight);
#endif


    for (IOWrap::Output3DWrapper *ow : outputWrapper) {
      ow->publishGraph(ef->connectivityMap);
      ow->publishKeyframes(frameHessians, false, &Hcalib);
    }



    // =========================== Marginalize Frames =========================
    //- When marginalize left frame, will also marginalize right frame.
    for (unsigned int i = 0; i < frameHessians.size(); i++) {
      if (frameHessians[i]->flaggedForMarginalization) {
        marginalizeFrame(frameHessians[i]);
        i = 0;
      }
    }


    printLogLine();
    //printEigenValLine();

  }


#if STEREO_MODE

  void FullSystem::initializeFromInitializerStereo(FrameHessian *newFrame) {
    boost::unique_lock<boost::mutex> lock(mapMutex);


    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();


    // add firstframe.
    FrameHessian *firstFrame = coarseInitializer->firstFrame;
    firstFrame->idx = frameHessians.size();
    frameHessians.push_back(firstFrame);
    firstFrame->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(firstFrame->shell);
    ef->insertFrame(firstFrame, &Hcalib);
    setPrecalcValues();

    FrameHessian *firstFrameRight = coarseInitializer->firstFrameRight;
    frameHessiansRight.push_back(firstFrameRight);

    firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

    float idepthStereo = 0;
    float sumID = 1e-5, numID = 1e-5;
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
      sumID += coarseInitializer->points[0][i].iR;
      numID++;
    }

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if (!setting_debugout_runquiet)
      printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
             (int) (setting_desiredPointDensity), coarseInitializer->numPoints[0]);

    //- initialize *first frame* by idepth computed by static stereo matching
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
      if (rand() / (float) RAND_MAX > keepPercentage) continue;

      Pnt *point = coarseInitializer->points[0] + i;
      ImmaturePoint *ip = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

      ip->u_stereo = ip->u;
      ip->v_stereo = ip->v;
      ip->idepth_min_stereo = 0;
      ip->idepth_max_stereo = NAN;

      const ImmaturePointStatus ipTraceRightStatus = ip->traceStereo(firstFrameRight, K, 1);

      if (ipTraceRightStatus == ImmaturePointStatus::IPS_GOOD) {
        ImmaturePoint *ipRight = new ImmaturePoint(ip->lastTraceUV(0), ip->lastTraceUV(1), firstFrameRight,
                                                   point->my_type, &Hcalib);

        ipRight->u_stereo = ipRight->u;
        ipRight->v_stereo = ipRight->v;
        ipRight->idepth_min_stereo = ipRight->idepth_min = 0;
        ipRight->idepth_max_stereo = ipRight->idepth_max = 0;
        const ImmaturePointStatus ipTraceLeftStatus = ipRight->traceStereo(firstFrame, K, 0);

        float u_stereo_delta = abs(ip->u_stereo - ipRight->lastTraceUV(0));
        float depth = 1.0f / ip->idepth_stereo;

        if (ipTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 &&
            depth < setting_acceptStaticDepthFactor * baseline) {
          ip->idepth_min = ip->idepth_min_stereo;
          ip->idepth_max = ip->idepth_max_stereo;

          PointHessian *ph = new PointHessian(ip, &Hcalib);
          delete ip;
          if (!std::isfinite(ph->energyTH)) {
            delete ph;
            continue;
          }

          ph->setIdepthScaled(ip->idepth_stereo);
          ph->setIdepthZero(ip->idepth_stereo);
          ph->hasDepthPrior = true;
          ph->setPointStatus(PointHessian::ACTIVE);


          firstFrame->pointHessians.push_back(ph);
          ef->insertPoint(ph);
        }
      }

      ip->idepth_min = ip->idepth_min_stereo;
      ip->idepth_max = ip->idepth_max_stereo;
      idepthStereo = ip->idepth_stereo;


      if (!std::isfinite(ip->energyTH) || !std::isfinite(ip->idepth_min) || !std::isfinite(ip->idepth_max)
          || ip->idepth_min < 0 || ip->idepth_max < 0) {
        delete ip;
        continue;

      }

      PointHessian *ph = new PointHessian(ip, &Hcalib);
      delete ip;
      if (!std::isfinite(ph->energyTH)) {
        delete ph;
        continue;
      }

      ph->setIdepthScaled(idepthStereo);
      ph->setIdepthZero(idepthStereo);
      ph->hasDepthPrior = true;
      ph->setPointStatus(PointHessian::ACTIVE);


      firstFrame->pointHessians.push_back(ph);
      ef->insertPoint(ph);
    }

    SE3 T_10 = coarseInitializer->thisToNext;

    // really no lock required, as we are initializing.
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      firstFrame->shell->T_WC = coarseInitializer->T_WC_ini; // already used IMU initialize this
      firstFrame->shell->aff_g2l = AffLight(0, 0);
      firstFrame->rightFrame->shell->aff_g2l = AffLight(0, 0);
      firstFrame->setEvalPT_scaled(firstFrame->shell->T_WC.inverse(),
                                   firstFrame->shell->aff_g2l,
                                   firstFrame->rightFrame->shell->aff_g2l);
      firstFrame->shell->trackingRef = 0;
      firstFrame->shell->camToTrackingRef = SE3();

//      firstFrameRight->shell->aff_g2l = firstFrame->shell->aff_g2l;
//      firstFrameRight->shell->T_WC = firstFrame->shell->T_WC * leftToRight_SE3.inverse();
//      firstFrameRight->setEvalPT_scaled(firstFrameRight->shell->T_WC.inverse(), firstFrameRight->shell->aff_g2l);

      newFrame->shell->T_WC = T_10.inverse();
      newFrame->shell->aff_g2l = AffLight(0, 0);
      newFrame->rightFrame->shell->aff_g2l = AffLight(0, 0);
      newFrame->setEvalPT_scaled(newFrame->shell->T_WC.inverse(),
                                 newFrame->shell->aff_g2l,
                                 newFrame->rightFrame->shell->aff_g2l);
      newFrame->shell->trackingRef = firstFrame->shell;
      newFrame->shell->camToTrackingRef = T_10.inverse();

//      newFrame->rightFrame->shell->aff_g2l = newFrame->shell->aff_g2l;
//      newFrame->rightFrame->shell->T_WC = newFrame->shell->T_WC * leftToRight_SE3.inverse();
//      newFrame->rightFrame->setEvalPT_scaled(newFrame->rightFrame->shell->T_WC.inverse(),
//                                             newFrame->rightFrame->shell->aff_g2l);
    }

    initialized = true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int) firstFrame->pointHessians.size());
  }

#else
  void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {

    boost::unique_lock<boost::mutex> lock(mapMutex);

    // add firstframe.
    FrameHessian *firstFrame = coarseInitializer->firstFrame;
    firstFrame->idx = frameHessians.size();
    frameHessians.push_back(firstFrame);
    frameHessiansRight.push_back(firstFrame->rightFrame);
    firstFrame->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(firstFrame->shell);
    ef->insertFrame(firstFrame, &Hcalib);
    setPrecalcValues();

    //int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    //int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

    firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);


    float sumID = 1e-5, numID = 1e-5;
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
      sumID += coarseInitializer->points[0][i].iR;
      numID++;
    }
    float rescaleFactor = 1 / (sumID / numID);

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if (!setting_debugout_runquiet)
      printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
             (int) (setting_desiredPointDensity), coarseInitializer->numPoints[0]);

    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
      if (rand() / (float) RAND_MAX > keepPercentage) continue;

      Pnt *point = coarseInitializer->points[0] + i;
      ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

      if (!std::isfinite(pt->energyTH)) {
        delete pt;
        continue;
      }


      pt->idepth_max = pt->idepth_min = 1;
      PointHessian *ph = new PointHessian(pt, &Hcalib);
      delete pt;
      if (!std::isfinite(ph->energyTH)) {
        delete ph;
        continue;
      }

      ph->setIdepthScaled(point->iR * rescaleFactor);
      ph->setIdepthZero(ph->idepth);
      ph->hasDepthPrior = true;
      ph->setPointStatus(PointHessian::ACTIVE);

      firstFrame->pointHessians.push_back(ph);
      ef->insertPoint(ph);
    }


    SE3 firstToNew = coarseInitializer->thisToNext;
    firstToNew.translation() /= rescaleFactor;


    // really no lock required, as we are initializing.
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      firstFrame->shell->T_WC = coarseInitializer->T_WC_ini;
      firstFrame->shell->aff_g2l = AffLight(0, 0);
      firstFrame->setEvalPT_scaled(firstFrame->shell->T_WC.inverse(), firstFrame->shell->aff_g2l);
      firstFrame->shell->trackingRef = 0;
      firstFrame->shell->camToTrackingRef = SE3();

      newFrame->shell->T_WC = firstToNew.inverse();
      newFrame->shell->aff_g2l = AffLight(0, 0);
      newFrame->setEvalPT_scaled(newFrame->shell->T_WC.inverse(), newFrame->shell->aff_g2l);
      newFrame->shell->trackingRef = firstFrame->shell;
      newFrame->shell->camToTrackingRef = firstToNew.inverse();

    }

    initialized = true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int) firstFrame->pointHessians.size());
  }
#endif

  void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth) {
    pixelSelector->allowFast = true;
    //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);
    int wl = wG[0];

    for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
      for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
        int i = x + y * wG[0];
        if (selectionMap[i] == 0) continue;

        ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
        if (!std::isfinite(impt->energyTH)) delete impt;
        else newFrame->immaturePoints.push_back(impt);

      }
    //printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

  }

#if STEREO_MODE
  void FullSystem::setPrecalcValues() {
    for (FrameHessian *fh : frameHessians) {
      fh->targetPrecalc.resize(frameHessians.size() + 1);
      for (unsigned int i = 0; i < frameHessians.size(); i++)
        fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
      fh->targetPrecalc.back().setStatic(fh, fh->rightFrame, &Hcalib);
    }
    ef->setDeltaF(&Hcalib);
  }
#else
  void FullSystem::setPrecalcValues() {
    for (FrameHessian *fh : frameHessians) {
      fh->targetPrecalc.resize(frameHessians.size());
      for (unsigned int i = 0; i < frameHessians.size(); i++)
        fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }
    ef->setDeltaF(&Hcalib);
  }
#endif


  void FullSystem::printLogLine() {
    if (frameHessians.size() == 0) return;

    if (!setting_debugout_runquiet)
      printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
             allKeyFramesHistory.back()->id,
             statistics_lastFineTrackRMSE,
             ef->resInA,
             ef->resInL,
             ef->resInM,
             (int) statistics_numForceDroppedResFwd,
             (int) statistics_numForceDroppedResBwd,
             allKeyFramesHistory.back()->aff_g2l.a,
             allKeyFramesHistory.back()->aff_g2l.b,
             frameHessians.back()->shell->id - frameHessians.front()->shell->id,
             (int) frameHessians.size());


    if (!setting_logStuff) return;

    if (numsLog != 0) {
      (*numsLog) << allKeyFramesHistory.back()->id << " " <<
                 statistics_lastFineTrackRMSE << " " <<
                 (int) statistics_numCreatedPoints << " " <<
                 (int) statistics_numActivatedPoints << " " <<
                 (int) statistics_numDroppedPoints << " " <<
                 (int) statistics_lastNumOptIts << " " <<
                 ef->resInA << " " <<
                 ef->resInL << " " <<
                 ef->resInM << " " <<
                 statistics_numMargResFwd << " " <<
                 statistics_numMargResBwd << " " <<
                 statistics_numForceDroppedResFwd << " " <<
                 statistics_numForceDroppedResBwd << " " <<
                 frameHessians.back()->aff_g2l().a << " " <<
                 frameHessians.back()->aff_g2l().b << " " <<
                 frameHessians.back()->shell->id - frameHessians.front()->shell->id << " " <<
                 (int) frameHessians.size() << " " << "\n";
      numsLog->flush();
    }


  }


  void FullSystem::printEigenValLine() {
    if (!setting_logStuff) return;
    if (ef->lastHS.rows() < 12) return;


    MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    int n = Hp.cols() / 8;
    assert(Hp.cols() % 8 == 0);

    // sub-select
    for (int i = 0; i < n; i++) {
      MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
      Hp.block(i * 6, 0, 6, n * 8) = tmp6;

      MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
      Ha.block(i * 2, 0, 2, n * 8) = tmp2;
    }
    for (int i = 0; i < n; i++) {
      MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
      Hp.block(0, i * 6, n * 8, 6) = tmp6;

      MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
      Ha.block(0, i * 2, n * 8, 2) = tmp2;
    }

    VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
    VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
    VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
    VecX diagonal = ef->lastHS.diagonal();

    std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
    std::sort(eigenP.data(), eigenP.data() + eigenP.size());
    std::sort(eigenA.data(), eigenA.data() + eigenA.size());

    int nz = std::max(100, setting_maxFrames * 10);

    if (eigenAllLog != 0) {
      VecX ea = VecX::Zero(nz);
      ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
      (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
      eigenAllLog->flush();
    }
    if (eigenALog != 0) {
      VecX ea = VecX::Zero(nz);
      ea.head(eigenA.size()) = eigenA;
      (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
      eigenALog->flush();
    }
    if (eigenPLog != 0) {
      VecX ea = VecX::Zero(nz);
      ea.head(eigenP.size()) = eigenP;
      (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
      eigenPLog->flush();
    }

    if (DiagonalLog != 0) {
      VecX ea = VecX::Zero(nz);
      ea.head(diagonal.size()) = diagonal;
      (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
      DiagonalLog->flush();
    }

    if (variancesLog != 0) {
      VecX ea = VecX::Zero(nz);
      ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
      (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
      variancesLog->flush();
    }

    std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
    (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
    for (unsigned int i = 0; i < nsp.size(); i++)
      (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
    (*nullspacesLog) << "\n";
    nullspacesLog->flush();

  }

  void FullSystem::printFrameLifetimes() {
    if (!setting_logStuff) return;


    boost::unique_lock<boost::mutex> lock(trackMutex);

    std::ofstream *lg = new std::ofstream();
    lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
    lg->precision(15);

    for (FrameShell *s : allFrameHistory) {
      (*lg) << s->id
            << " " << s->marginalizedAt
            << " " << s->statistics_goodResOnThis
            << " " << s->statistics_outlierResOnThis
            << " " << s->movedByOpt;


      (*lg) << "\n";
    }


    lg->close();
    delete lg;

  }


  void FullSystem::printEvalLine() {
    return;
  }


}
