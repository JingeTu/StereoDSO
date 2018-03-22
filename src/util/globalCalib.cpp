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



#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

namespace dso {
  int wG[PYR_LEVELS], hG[PYR_LEVELS];
  float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
      cxG[PYR_LEVELS], cyG[PYR_LEVELS];

  float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
      cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

  Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

  float baseline;
  SE3 T_SC0;
  IMUParameters imuParameters;
  Mat66 d_xi_d_xi_c;

  float wM3G;
  float hM3G;

  void setGlobalCameraCalib(int w, int h, const Eigen::Matrix3f &K) {
    int wlvl = w;
    int hlvl = h;
    pyrLevelsUsed = 1;
    while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS) {
      wlvl /= 2;
      hlvl /= 2;
      pyrLevelsUsed++;
    }
    printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
           pyrLevelsUsed - 1, wlvl, hlvl);
    if (wlvl > 100 && hlvl > 100) {
      printf("\n\n===============WARNING!===================\n "
                 "using not enough pyramid levels.\n"
                 "Consider scaling to a resolution that is a multiple of a power of 2.\n");
    }
    if (pyrLevelsUsed < 3) {
      printf("\n\n===============WARNING!===================\n "
                 "I need higher resolution.\n"
                 "I will probably segfault.\n");
    }

    wM3G = w - 3;
    hM3G = h - 3;

    wG[0] = w;
    hG[0] = h;
    KG[0] = K;
    fxG[0] = K(0, 0);
    fyG[0] = K(1, 1);
    cxG[0] = K(0, 2);
    cyG[0] = K(1, 2);
    KiG[0] = KG[0].inverse();
    fxiG[0] = KiG[0](0, 0);
    fyiG[0] = KiG[0](1, 1);
    cxiG[0] = KiG[0](0, 2);
    cyiG[0] = KiG[0](1, 2);

    for (int level = 1; level < pyrLevelsUsed; ++level) {
      wG[level] = w >> level;
      hG[level] = h >> level;

      fxG[level] = fxG[level - 1] * 0.5;
      fyG[level] = fyG[level - 1] * 0.5;
      cxG[level] = (cxG[0] + 0.5) / ((int) 1 << level) - 0.5;
      cyG[level] = (cyG[0] + 0.5) / ((int) 1 << level) - 0.5;

      KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;  // synthetic
      KiG[level] = KG[level].inverse();

      fxiG[level] = KiG[level](0, 0);
      fyiG[level] = KiG[level](1, 1);
      cxiG[level] = KiG[level](0, 2);
      cyiG[level] = KiG[level](1, 2);
    }
  }

  void setGlobalIMUCalib() {

    Mat44 T;
    T << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0;
    T_SC0.setRotationMatrix(T.topLeftCorner(3, 3));
    T_SC0.translation() = T.topRightCorner(3, 1);

    d_xi_d_xi_c = T_SC0.Adj();

    imuParameters.a_max = 176.0; // # acceleration saturation [m/s^2]
    imuParameters.g_max = 7.8; // # gyro saturation [rad/s]
    imuParameters.sigma_g_c = 12.0e-4; // # gyro noise density [rad/s/sqrt(Hz)]
    imuParameters.sigma_a_c = 8.0e-3; // # accelerometer noise density [m/s^2/sqrt(Hz)]
    imuParameters.sigma_bg = 0.03; // # gyro bias prior [rad/s]
    imuParameters.sigma_ba = 0.1; // # accelerometer bias prior [m/s^2]
    imuParameters.sigma_gw_c = 4.0e-6; // # gyro drift noise density [rad/s^s/sqrt(Hz)]
    imuParameters.sigma_aw_c = 4.0e-5; // # accelerometer drift noise density [m/s^2/sqrt(Hz)]
    imuParameters.tau = 3600.0; // # reversion time constant, currently not in use [s]
    imuParameters.g = 9.81007; // # Earth's acceleration due to gravity [m/s^2]
    imuParameters.a0 << 0.0, 0.0, 0.0; // # Accelerometer bias [m/s^2]
    imuParameters.rate = 200;
    imuParameters.T_BS = SE3();
  }


}
