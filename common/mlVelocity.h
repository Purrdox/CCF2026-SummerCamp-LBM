#pragma once

#ifndef _MLVELOCITY_
#define _MLVELOCITY_

#include "mlCoreWin.h"

template<class T>
struct mlVelocity2D
{
	T ux;
	T uy;

	MLFUNC_TYPE T getMag();

	MLFUNC_TYPE mlVelocity2D<T> & operator = (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator += (const mlVelocity2D<T> &u);
	MLFUNC_TYPE mlVelocity2D<T>  operator +  (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator -= (const mlVelocity2D<T> &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator -  (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator *= (const mlVelocity2D<T> &u);
	MLFUNC_TYPE mlVelocity2D<T>  operator *  (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator /= (const mlVelocity2D<T> &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator /  (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator += (const T &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator +  (const T &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator -= (const T &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator -  (const T &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator *= (const T &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator *  (const T &u);

	MLFUNC_TYPE mlVelocity2D<T> & operator /= (const T &u);
	MLFUNC_TYPE mlVelocity2D<T>   operator /  (const T &u);

	MLFUNC_TYPE bool   operator ==  (const mlVelocity2D<T> &u);
	MLFUNC_TYPE bool   operator !=  (const mlVelocity2D<T> &u);

	MLFUNC_TYPE mlVelocity2D(const T &ux, const T &uy);
	MLFUNC_TYPE mlVelocity2D(const T val);
	MLFUNC_TYPE mlVelocity2D(const mlVelocity2D<T> &u);
	MLFUNC_TYPE mlVelocity2D();
};

typedef mlVelocity2D<REAL> mlVelocity2f;
typedef mlVelocity2D<REAL> mlVelocity2d;

///////////////implementation//////////////

template<class T>
MLFUNC_TYPE mlVelocity2D<T>::mlVelocity2D()
{
	ux = uy = T(0.0);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>::mlVelocity2D(const T &ux, const T &uy)
{
	this->ux = ux;
	this->uy = uy;
}

template<class T>
inline mlVelocity2D<T>::mlVelocity2D(const T val)
{
	ux = uy = val;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>::mlVelocity2D(const mlVelocity2D<T> &u)
{
	ux = u.ux;
	uy = u.uy;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator = (const mlVelocity2D<T> &u)
{
	ux = u.ux;
	uy = u.uy;

	return (*this);
}

template<class T>
MLFUNC_TYPE T mlVelocity2D<T>::getMag()
{
	return (T)sqrt(double(ux*ux + uy*uy));
	//return sqrt(double(ux*ux + uy*uy));
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator += (const mlVelocity2D<T> &u)
{
	ux += u.ux;
	uy += u.uy;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator +  (const mlVelocity2D<T> &u)
{
	mlVelocity2D<T> temp(*this);
	return temp += u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator -= (const mlVelocity2D<T> &u)
{
	ux -= u.ux;
	uy -= u.uy;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator -  (const mlVelocity2D<T> &u)
{
	mlVelocity2D<T> temp(*this);
	return temp -= u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator *= (const mlVelocity2D<T> &u)
{
	ux *= u.ux;
	uy *= u.uy;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator *  (const mlVelocity2D<T> &u)
{
	mlVelocity2D<T> temp(*this);
	return temp *= u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator /= (const mlVelocity2D<T> &u)
{
	ux /= u.ux;
	uy /= u.uy;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator /  (const mlVelocity2D<T> &u)
{
	mlVelocity2D<T> temp(*this);
	return temp /= u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator += (const T &u)
{
	ux += u;
	uy += u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator +  (const T &u)
{
	mlVelocity2D<T> temp(*this);
	return temp += u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator -= (const T &u)
{
	ux -= u;
	uy -= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator -  (const T &u)
{
	mlVelocity2D<T> temp(*this);
	return temp -= u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator *= (const T &u)
{
	ux *= u;
	uy *= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator *  (const T &u)
{
	mlVelocity2D<T> temp(*this);
	return temp *= u;
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T> & mlVelocity2D<T>::operator /= (const T &u)
{
	ux /= u;
	uy /= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity2D<T>   mlVelocity2D<T>::operator /  (const T &u)
{
	mlVelocity2D<T> temp(*this);
	return temp /= u;
}

template<class T>
MLFUNC_TYPE bool   mlVelocity2D<T>::operator ==  (const mlVelocity2D<T> &u)
{
	if (ux == u.ux && uy == u.uy)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool   mlVelocity2D<T>::operator !=  (const mlVelocity2D<T> &u)
{
	return !((*this) == u);
}


////////////////////////////////////////////////////

template<class T>
struct mlVelocity3D
{
	T ux;
	T uy;
	T uz;

	MLFUNC_TYPE T getMag();

	MLFUNC_TYPE mlVelocity3D<T> & operator = (const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D<T> & operator = (const T &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator += (const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D<T> & operator +  (const mlVelocity3D<T> &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator -= (const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D<T> & operator -  (const mlVelocity3D<T> &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator *= (const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D<T> & operator *  (const mlVelocity3D<T> &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator /= (const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D<T> & operator /  (const mlVelocity3D<T> &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator += (const T &u);
	MLFUNC_TYPE mlVelocity3D<T>   operator +  (const T &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator -= (const T &u);
	MLFUNC_TYPE mlVelocity3D<T>   operator -  (const T &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator *= (const T &u);
	MLFUNC_TYPE mlVelocity3D<T>   operator *  (const T &u);

	MLFUNC_TYPE mlVelocity3D<T> & operator /= (const T &u);
	MLFUNC_TYPE mlVelocity3D<T>   operator /  (const T &u);

	MLFUNC_TYPE bool   operator ==  (const mlVelocity3D<T> &u);
	MLFUNC_TYPE bool   operator !=  (const mlVelocity3D<T> &u);

	MLFUNC_TYPE mlVelocity3D(const T &ux, const T &uy, const T &uz);
	MLFUNC_TYPE mlVelocity3D(const mlVelocity3D<T> &u);
	MLFUNC_TYPE mlVelocity3D(const T &u);
	MLFUNC_TYPE mlVelocity3D();
	MLFUNC_TYPE mlVelocity3D<T> CrossProduct(mlVelocity3D<T> &u);
};

typedef mlVelocity3D<REAL> mlVelocity3f;
typedef mlVelocity3D<double> mlVelocity3d;

///////////////inline implementation//////////////

template<class T>
MLFUNC_TYPE mlVelocity3D<T>::mlVelocity3D()
{
	ux = uy = uz = T(0.0);
}

template<class T>
inline MLFUNC_TYPE mlVelocity3D<T> mlVelocity3D<T>::CrossProduct(mlVelocity3D<T>& u)
{
	T  x = this->uy*u.uz - this->uz*u.uy;
	T  y = this->uz*u.ux - this->ux*u.uz;
	T  z = this->ux*u.uy - this->uy*u.ux;
	return mlVelocity3D<T>(x, y, z);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>::mlVelocity3D(const T &ux, const T &uy, const T &uz)
{
	this->ux = ux;
	this->uy = uy;
	this->uz = uz;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>::mlVelocity3D(const mlVelocity3D<T> &u)
{
	ux = u.ux;
	uy = u.uy;
	uz = u.uz;
}

template<class T>
inline mlVelocity3D<T>::mlVelocity3D(const T & u)
{
	ux = uy = uz = u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator = (const mlVelocity3D<T> &u)
{
	ux = u.ux;
	uy = u.uy;
	uz = u.uz;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator = (const T &u)
{
	ux = u;
	uy = u;
	uz = u;

	return (*this);
}

template<class T>
MLFUNC_TYPE T mlVelocity3D<T>::getMag()
{
	return (T)sqrt(double(ux*ux + uy*uy + uz*uz));
	//return sqrt(double(ux*ux + uy*uy + uz*uz));
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator += (const mlVelocity3D<T> &u)
{
	ux += u.ux;
	uy += u.uy;
	uz += u.uz;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator +  (const mlVelocity3D<T> &u)
{
	mlVelocity3D<T> temp(*this);
	return temp += u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator -= (const mlVelocity3D<T> &u)
{
	ux -= u.ux;
	uy -= u.uy;
	uz -= u.uz;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator -  (const mlVelocity3D<T> &u)
{
	mlVelocity3D<T> temp(*this);
	return temp -= u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator *= (const mlVelocity3D<T> &u)
{
	ux *= u.ux;
	uy *= u.uy;
	uz *= u.uz;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator *  (const mlVelocity3D<T> &u)
{
	mlVelocity3D<T> temp(*this);
	return temp *= u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator /= (const mlVelocity3D<T> &u)
{
	ux /= u.ux;
	uy /= u.uy;
	uz /= u.uz;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator /  (const mlVelocity3D<T> &u)
{
	mlVelocity3D<T> temp(*this);
	return temp /= u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator += (const T &u)
{
	ux += u;
	uy += u;
	uz += u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>   mlVelocity3D<T>::operator +  (const T &u)
{
	mlVelocity3D<T> temp(*this);
	return temp += u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator -= (const T &u)
{
	ux -= u;
	uy -= u;
	uz -= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>   mlVelocity3D<T>::operator -  (const T &u)
{
	mlVelocity3D<T> temp(*this);
	return temp -= u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator *= (const T &u)
{
	ux *= u;
	uy *= u;
	uz *= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>   mlVelocity3D<T>::operator *  (const T &u)
{
	mlVelocity3D<T> temp(*this);
	return temp *= u;
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T> & mlVelocity3D<T>::operator /= (const T &u)
{
	ux /= u;
	uy /= u;
	uz /= u;

	return (*this);
}

template<class T>
MLFUNC_TYPE mlVelocity3D<T>   mlVelocity3D<T>::operator /  (const T &u)
{
	mlVelocity3D<T> temp(*this);
	return temp /= u;
}

template<class T>
MLFUNC_TYPE bool   mlVelocity3D<T>::operator ==  (const mlVelocity3D<T> &u)
{
	if (ux == u.ux && uy == u.uy && uz == u.uz)
		return true;
	else
		return false;
}

template<class T>
MLFUNC_TYPE bool   mlVelocity3D<T>::operator !=  (const mlVelocity3D<T> &u)
{
	return !((*this) == u);
}


#endif //_MLVELOCITY_