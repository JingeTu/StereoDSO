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
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"


namespace dso {
  struct CalibHessian;
  struct FrameHessian;
  struct PointFrameResidual;

  class CoarseTracker {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseTracker(int w, int h);

    ~CoarseTracker();

#if STEREO_MODE

    bool trackNewestCoarseStereo(
        FrameHessian *newFrameHessian,
        FrameHessian *newFrameHessianRight,
        SE3 &lastToNew_out,
        AffLight &aff_g2l_out, AffLight &aff_g2l_r_out,
        int coarsestLvl, Vec5 minResForAbort,
        IOWrap::Output3DWrapper *wrap = 0);

#endif
#if !STEREO_MODE & !INERTIAL_MODE

    bool trackNewestCoarse(
        FrameHessian *newFrameHessian,
        SE3 &lastToNew_out, AffLight &aff_g2l_out,
        int coarsestLvl, Vec5 minResForAbort,
        IOWrap::Output3DWrapper *wrap = 0);

#endif

    void setCTRefForFirstFrame(std::vector<FrameHessian *> frameHessians);

    void makeCoarseDepthForFirstFrame(FrameHessian *fh);

    void setCoarseTrackingRef(
        std::vector<FrameHessian *> frameHessians);

    void makeK(
        CalibHessian *HCalib);

    bool debugPrint, debugPlot;

    Mat33f K[PYR_LEVELS];
    Mat33f Ki[PYR_LEVELS];
    float fx[PYR_LEVELS];
    float fy[PYR_LEVELS];
    float fxi[PYR_LEVELS];
    float fyi[PYR_LEVELS];
    float cx[PYR_LEVELS];
    float cy[PYR_LEVELS];
    float cxi[PYR_LEVELS];
    float cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

    void debugPlotIDepthMap(float *minID, float *maxID, std::vector<IOWrap::Output3DWrapper *> &wraps);

    void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps);

    FrameHessian *lastRef;
    AffLight lastRef_aff_g2l;
    FrameHessian *newFrame;
    FrameHessian *newFrameRight;
    int refFrameID;

    //- This is jinge comment
    /*
     * lastResiduals for 5 level image alignment.
     * Each is calculated as lastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));
     * Its reprojection photometric error average.
     */
    // act as pure ouptut
    Vec5 lastResiduals;
    Vec3 lastFlowIndicators;
    double firstCoarseRMSE;
  private:


    void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);

    float *idepth[PYR_LEVELS];
    float *weightSums[PYR_LEVELS];
    float *weightSums_bak[PYR_LEVELS];


    Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);

#if STEREO_MODE

    Vec6 calcResStereo(int lvl, const SE3 &refToNew, AffLight aff_g2l, AffLight aff_g2l_r, float cutoffTH);

    void
    calcGSSSEStereo(int lvl, Mat1010 &H_out, Vec10 &b_out, const SE3 &refToNew, AffLight aff_g2l, AffLight aff_g2l_r);

#endif
#if !STEREO_MODE & !INERTIAL_MODE

    Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);

    void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

#endif

    void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

    // pc buffers
    float *pc_u[PYR_LEVELS];
    float *pc_v[PYR_LEVELS];
    float *pc_idepth[PYR_LEVELS];
    float *pc_color[PYR_LEVELS];
    int pc_n[PYR_LEVELS];

    // warped buffers
    float *buf_warped_idepth;
    float *buf_warped_u;
    float *buf_warped_v;
    float *buf_warped_dx;
    float *buf_warped_dy;
    float *buf_warped_residual;
    float *buf_warped_weight;
    float *buf_warped_refColor;
    int buf_warped_n;
    //- warped buffers for stereo
    float *buf_warped_idepth_r;
    float *buf_warped_dx_r;
    float *buf_warped_dy_r;
    float *buf_warped_residual_r;
    float *buf_warped_weight_r;


    std::vector<float *> ptrToDelete;

#if STEREO_MODE
    Accumulator11 acc;
#endif
#if !STEREO_MODE & !INERTIAL_MODE
    Accumulator9 acc;
#endif
  };


  class CoarseDistanceMap {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseDistanceMap(int w, int h);

    ~CoarseDistanceMap();

    void makeDistanceMap(
        std::vector<FrameHessian *> frameHessians,
        FrameHessian *frame);

    void makeInlierVotes(
        std::vector<FrameHessian *> frameHessians);

    void makeK(CalibHessian *HCalib);


    float *fwdWarpedIDDistFinal;

    Mat33f K[PYR_LEVELS];
    Mat33f Ki[PYR_LEVELS];
    float fx[PYR_LEVELS];
    float fy[PYR_LEVELS];
    float fxi[PYR_LEVELS];
    float fyi[PYR_LEVELS];
    float cx[PYR_LEVELS];
    float cy[PYR_LEVELS];
    float cxi[PYR_LEVELS];
    float cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

    void addIntoDistFinal(int u, int v);


  private:

    PointFrameResidual **coarseProjectionGrid;
    int *coarseProjectionGridNum;
    Eigen::Vector2i *bfsList1;
    Eigen::Vector2i *bfsList2;

    void growDistBFS(int bfsNum);
  };

}

