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


#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {


  void EFResidual::takeDataF() {
    std::swap<RawResidualJacobian *>(J, data->J);

    Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

    for (int i = 0; i < 6; i++)
      JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];

//    if (targetIDX == -1) { //-- checkout static residual
//      LOG(INFO) << "J->JabJIdx: " << J->JabJIdx;
//      LOG(INFO) << "J->Jpdd: " << J->Jpdd;
//      LOG(INFO) << "\t";
//    }

#if defined(STEREO_MODE)
    JpJdF.segment<4>(6) = J->JabJIdx * J->Jpdd;
#else
    JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
#endif
  }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  void EFIMUResidual::takeDataF() {
    std::swap<RawIMUResidualJacobian *>(J, data->J);
    // [0] xi0, s0, [1] xi0, s1, [2] xi1, s0, [3] xi1, s1
    JxiJsF[0] = (J->Jrdxi[0].transpose() * J->Jrdsb[0]).cast<float>();
    JxiJsF[1] = (J->Jrdxi[0].transpose() * J->Jrdsb[1]).cast<float>();
    JxiJsF[2] = (J->Jrdxi[1].transpose() * J->Jrdsb[0]).cast<float>();
    JxiJsF[3] = (J->Jrdxi[1].transpose() * J->Jrdsb[1]).cast<float>();
  }

#endif

  void EFFrame::takeData() {
#if defined(STEREO_MODE)
    prior = data->getPrior().head<10>();
    delta = data->get_state_minus_stateZero().head<10>();
    delta_prior = (data->get_state() - data->getPriorZero()).head<10>();
#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)
    prior = data->getPrior().head<8>();
    delta = data->get_state_minus_stateZero().head<8>();
    delta_prior = (data->get_state() - data->getPriorZero()).head<8>();
#endif


//	Vec10 state_zero =  data->get_state_zero();
//	state_zero.segment<3>(0) = SCALE_XI_TRANS * state_zero.segment<3>(0);
//	state_zero.segment<3>(3) = SCALE_XI_ROT * state_zero.segment<3>(3);
//	state_zero[6] = SCALE_A * state_zero[6];
//	state_zero[7] = SCALE_B * state_zero[7];
//	state_zero[8] = SCALE_A * state_zero[8];
//	state_zero[9] = SCALE_B * state_zero[9];
//
//	std::cout << "state_zero: " << state_zero.transpose() << "\n";


//    assert(data->frameID != -1);

    frameID = data->frameID;
  }


  void EFPoint::takeData() {
    priorF = data->hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR) priorF = 0;

    deltaF = data->idepth - data->idepth_zero;
  }

#if defined(STEREO_MODE) && defined(INERTIAL_MODE)

  void EFSpeedAndBias::takeData() {
    priorF.setZero();
    deltaF = data->state - data->state_zero;
  }

  void EFIMUResidual::fixLinearizationF(EnergyFunctional *ef) {
    SpeedAndBiasf dsbf = this->from_sb->data->get_state_minus_stateZero().cast<float>();
    SpeedAndBiasf dsbt = this->to_sb->data->get_state_minus_stateZero().cast<float>();
    Vec6f dxif = this->from_f->data->get_state_minus_stateZero().head<6>().cast<float>();
    Vec6f dxit = this->to_f->data->get_state_minus_stateZero().head<6>().cast<float>();

    res_toZeroF = J->Jrdsb[0] * dsbf + J->Jrdsb[1] * dsbt
                  + J->Jrdxi[0] * dxif + J->Jrdxi[1] * dxit;
  }

#endif

#if defined(STEREO_MODE)

  void EFResidual::fixLinearizationF(EnergyFunctional *ef) {
    Vec10f dp;
    if (targetIDX == -1) { //- static stereo residual
      dp = ef->adHTdeltaF[hostIDX + ef->nFrames * hostIDX];
    }
    else { //- temporal stereo residual
      dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];
    }

    // compute Jp*delta
    __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                    + J->Jpdc[0].dot(ef->cDeltaF)
                                    + J->Jpdd[0] * point->deltaF);
    __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                    + J->Jpdc[1].dot(ef->cDeltaF)
                                    + J->Jpdd[1] * point->deltaF);
    __m128 delta_a = _mm_set1_ps((float) (dp[6]));
    __m128 delta_b = _mm_set1_ps((float) (dp[7]));
    __m128 delta_a_r = _mm_set1_ps((float) (dp[8]));
    __m128 delta_b_r = _mm_set1_ps((float) (dp[9]));

    for (int i = 0; i < patternNum; i += 4) {
      // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
      __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 2)) + i), delta_a_r));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 3)) + i), delta_b_r));
      _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
    }

    isLinearized = true;
  }

#endif
#if !defined(STEREO_MODE) && !defined(INERTIAL_MODE)

  void EFResidual::fixLinearizationF(EnergyFunctional *ef) {
    Vec8f dp;
    if (targetIDX == -1) { //- static stereo residual
      dp = ef->adHTdeltaF[hostIDX + ef->nFrames * hostIDX];
    }
    else { //- temporal stereo residual
      dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];
    }

    // compute Jp*delta
    __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                    + J->Jpdc[0].dot(ef->cDeltaF)
                                    + J->Jpdd[0] * point->deltaF);
    __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                    + J->Jpdc[1].dot(ef->cDeltaF)
                                    + J->Jpdd[1] * point->deltaF);
    __m128 delta_a = _mm_set1_ps((float) (dp[6]));
    __m128 delta_b = _mm_set1_ps((float) (dp[7]));

    for (int i = 0; i < patternNum; i += 4) {
      // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
      __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
      rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
      _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
    }

    isLinearized = true;
  }

#endif
}
