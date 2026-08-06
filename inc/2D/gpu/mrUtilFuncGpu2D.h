#pragma once
#ifndef _MRUTILFUNCGU2DH_
#define _MRUTILFUNCGU2DH_

#include "cuda_runtime.h"
//#include "../../../lw_core_win/mlCoreWinHeader.h"
#include "../../../common/mlCoreWin.h"
#include "../../../common/mlLatticeNode.h"

#include "mrConstantParamsGpu2D.h"
class mrUtilFuncGpu2D
{
public:
	MLFUNC_TYPE void mlCalDistributionD2Q9AtIndex(
		REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out);
	MLFUNC_TYPE void mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega,
		REAL& pixx,
		REAL& piyy,
		REAL& pixy);
 

private:

};

inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlCalDistributionD2Q9AtIndex(
	REAL rho, REAL ux, REAL uy, REAL pixx, REAL pixy, REAL piyy, int i, REAL& f_out)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-A D2Q9 reconstruction =====
	// TODO: Reconstruct population f_i from density, velocity, and
	// normalized second moments. Build the required moment/Hermite
	// coefficients, then evaluate the polynomial for direction i with the
	// corresponding D2Q9 weight.
#ifdef __CUDA_ARCH__
	const REAL a0 = rho;
	const REAL ax = rho * ux;
	const REAL ay = rho * uy;
	const REAL axx = rho * pixx;
	const REAL ayy = rho * piyy;
	const REAL axy = rho * pixy;
	const REAL axxy =
		-2.0f * rho * uy * ux * ux + 2.0f * axy * ux + axx * uy;
	const REAL axyy =
		-2.0f * rho * ux * uy * uy + 2.0f * axy * uy + ayy * ux;

	REAL polynomial = 0.0f;
	switch (i)
	{
	case 0:
		polynomial = a0 - 1.5f * axx - 1.5f * ayy;
		break;
	case 1:
		polynomial = a0 + 3.0f * ax + 3.0f * axx - 4.5f * axyy - 1.5f * ayy;
		break;
	case 2:
		polynomial = a0 - 1.5f * axx - 4.5f * axxy + 3.0f * ay + 3.0f * ayy;
		break;
	case 3:
		polynomial = a0 - 3.0f * ax + 3.0f * axx + 4.5f * axyy - 1.5f * ayy;
		break;
	case 4:
		polynomial = a0 - 1.5f * axx + 4.5f * axxy - 3.0f * ay + 3.0f * ayy;
		break;
	case 5:
		polynomial = a0 + 3.0f * ax + 3.0f * axx + 9.0f * axy +
			9.0f * axxy + 9.0f * axyy + 3.0f * ay + 3.0f * ayy;
		break;
	case 6:
		polynomial = a0 - 3.0f * ax + 3.0f * axx - 9.0f * axy +
			9.0f * axxy - 9.0f * axyy + 3.0f * ay + 3.0f * ayy;
		break;
	case 7:
		polynomial = a0 - 3.0f * ax + 3.0f * axx + 9.0f * axy -
			9.0f * axxy - 9.0f * axyy - 3.0f * ay + 3.0f * ayy;
		break;
	case 8:
		polynomial = a0 + 3.0f * ax + 3.0f * axx - 9.0f * axy -
			9.0f * axxy + 9.0f * axyy - 3.0f * ay + 3.0f * ayy;
		break;
	}

	f_out = w2d_gpu[i] * polynomial;
	if (f_out < 0.0f)
	{
		f_out = 0.0f;
	}
#else
	f_out = 0.0f;
#endif
	// ===== ASSIGNMENT FILL END: P1-A =====
}

inline MLFUNC_TYPE void mrUtilFuncGpu2D::mlGetPIAfterCollision(REAL R, REAL U, REAL V, REAL Fx, REAL Fy, REAL omega, REAL& pixx, REAL& piyy, REAL& pixy)
{
	// ===== ASSIGNMENT FILL BEGIN: P1-B second-moment collision =====
	// TODO: Apply the BGK relaxation and body-force correction to the
	// three raw second moments in place. The equilibrium contribution must be
	// consistent with cs^2 = 1/3 and the supplied macroscopic velocity.
#ifdef __CUDA_ARCH__
	const REAL pixx_eq = R * (U * U + cs2);
	const REAL piyy_eq = R * (V * V + cs2);
	const REAL pixy_eq = R * U * V;

	pixx += omega * (pixx_eq - pixx);
	piyy += omega * (piyy_eq -piyy);
	pixy += omega * (pixy_eq - pixy);

	const REAL forceFactor = 1.0f - 0.5f * omega;
	pixx += forceFactor * 2.0f * U * Fx;
	piyy += forceFactor * 2.0f * V * Fy;
	pixy += forceFactor * (U * Fy + V * Fx);
#endif
	// ===== ASSIGNMENT FILL END: P1-B =====
}

 

#endif // !_MRUTILFUNCGU2DH_
