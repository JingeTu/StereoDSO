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
#include "util/settings.h"

namespace dso {
  struct RawResidualJacobian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // ================== new structure: save independently =============.
    VecNRf resF; // typedef Eigen::Matrix<float,MAX_RES_PER_POINT,1> VecNRf; MAX_RES_PER_POINT == 8

    // the two rows of d[x,y]/d[xi].
    Vec6f Jpdxi[2];      // 2x6

    // the two rows of d[x,y]/d[C].
    VecCf Jpdc[2];      // 2x4

    // the two rows of d[x,y]/d[idepth].
    Vec2f Jpdd;        // 2x1

    // the two columns of d[r]/d[x,y].
    VecNRf JIdx[2];      // 8x2

#if STEREO_MODE
    // = the two columns of d[r] / d[ab]. Includes rightFrame ab.
    VecNRf JabF[4];      // 8x4
#else
    // = the two columns of d[r] / d[ab]
    VecNRf JabF[2];      // 8x2
#endif

    // = JIdx^T * JIdx (inner product). Only as a shorthand.
    Mat22f JIdx2;        // 2x2
#if STEREO_MODE
    //- = Jab^T * JIdx (innter product). Only as a shorhand. Includes rightFrame ab.
    Mat42f JabJIdx;
#else
    // = Jab^T * JIdx (inner product). Only as a shorthand.
    Mat22f JabJIdx;      // 2x2
#endif
#if STEREO_MODE
    // = Jab^T * Jab (inner product). Only as a shorthand. Includes rightFrame ab.
    Mat44f Jab2;      // 4x4
#else
    // = Jab^T * Jab (inner product). Only as a shorthand.
    Mat22f Jab2;      // 2x2
#endif
  };

#if STEREO_MODE & INERTIAL_MODE
  struct RawIMUResidualJacobian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<float, 15, 1> resF; //- residual

    Eigen::Matrix<float, 15, 6> Jrdxi[2]; //- Derivative with respect to host & target pose

    Eigen::Matrix<float, 15, 9> Jrdsb[2]; //- Derivative with respect to host & target SpeedAndBiases
  };
#endif
}

