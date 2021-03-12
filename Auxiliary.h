#pragma once
#include "gaussians.h"
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include "toms462.h"

using namespace Eigen;

using namespace boost::math::quadrature;


inline MatrixXd matrix_exponential(const MatrixXd input_matrix,size_t itr)
{
	size_t size = input_matrix.rows();
	MatrixXd res1 = Matrix3d::Identity(size, size);
	MatrixXd res2 = res1;
	for (size_t i = 0; i < size; i++)
	{
		res2 = res2 * (input_matrix/((double)(i)+1.0));
		res1 = res1 + res2;
	}
	return res1;
}

inline double AFNS3_yield_adjustment(const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const double time)
{
	const double A = sigma11*sigma11 + sigma12*sigma12 + sigma13*sigma13;

	const double B = sigma21*sigma21 + sigma22*sigma22 + sigma23*sigma23;

	const double C = sigma31*sigma31 + sigma32*sigma32 + sigma33*sigma33;

	const double D = sigma11 * sigma21 + sigma12 * sigma22 + sigma13 * sigma23;

	const double E = sigma11 * sigma31 + sigma12 * sigma32 + sigma13 * sigma33;

	const double F = sigma21 * sigma31 + sigma22 * sigma32 + sigma23 * sigma33;

	const double e1 = exp(-lambda * time);
	
	const double e2 = exp(-2.0 * lambda*time);

	const double l3 = lambda*lambda*lambda*time;

	const double sq_lambda = lambda * lambda;

	const double r1 = -A * time*time / 6.0;

	const double r2 = -B * (1.0 / (2.0 * sq_lambda) - (1.0 -e1) / l3 + (1.0 - e2) / (4.0 * l3));

	const double r3 = -C * (1.0 / (2.0 * sq_lambda) + e1 / sq_lambda - time * e2 / (4.0 * lambda) - 3.0 *e2 / (4.0*sq_lambda)- 2.0*(1.0-e1)/l3 + 5.0 * (1.0 - e2) / (8.0 * l3));

	const double r4 = -D * (time / (2.0 * lambda) + e1 / (sq_lambda) - (1.0 -e1) / l3);

	const double r5 = -E * (3.0*e1 / sq_lambda + time / (2.0 * lambda) + time * e1 / lambda - 3.0*(1.0 - e1) / l3);

	const double r6 = -F * (1.0 / sq_lambda + e1 / sq_lambda - e2 / (2.0 * sq_lambda) - 3.0*(1.0 - e1) / l3 + 3.0*(1.0 - e2) / (4.0*l3));

	return r1+r2+r3+r4+r5+r6;
}

inline double AFNS3_A(const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const double time)
{
	
	return -AFNS3_yield_adjustment(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, time)*time;

}

inline MatrixXd AFNS3_B2(const double lambda, const double Maturity)
{
	VectorXd B(3);

		B(0) = -Maturity;
		B(1) = -1.0/(lambda)*(1.0 - exp(-lambda * Maturity));
		B(2) = B(1) +Maturity*exp(-lambda *Maturity);

	return B;
}

inline MatrixXd AFNS3_B3(const double lambda, const double start, const double end)
{
	VectorXd B(3);

	B(0) = -(end-start);
	B(1) = -1.0 / (lambda) * (1.0 - exp(-lambda * (end - start)));
	B(2) = B(1) + (end - start) * exp(-lambda * (end - start));

	return B;
}


inline double ZCB_AFNS(const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const double time, const VectorXd X)
{
	double A = AFNS3_A( sigma11,  sigma12,  sigma13,  sigma21,  sigma22,  sigma23,  sigma31,  sigma32,  sigma33,  lambda,  time);

	VectorXd B = AFNS3_B2(lambda, time);

	return exp(A + B.transpose()*X);
}

inline MatrixXd fut_price_1m(const MatrixXd Data, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{
	//VectorXd rates(Data.rows());
	MatrixXd rates_derivs(Data.rows(), 4);

	Matrix3d Sigma;
	Sigma << sigma11, sigma12, sigma13,
		sigma21, sigma22, sigma23,
		sigma31, sigma32, sigma33;

	Matrix3d Sigmasq = Sigma * Sigma.transpose();


	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{
			Vector3d B_end = AFNS3_B2(lambda, dt * Data(i, 3));
			Vector3d B = (-B_end) / (Data(i, 1) * dt);
			double rate = Data(i, 4) + B.dot(X);
			rates_derivs.row(i) << rate, B.transpose();

			
		}
		else
		{

			Vector3d B_start2 = AFNS3_B2(lambda, dt * Data(i, 2));
			Vector3d B_end2 = AFNS3_B2(lambda, dt * Data(i, 3));
			Vector3d B2 = (B_start2 - B_end2) / (Data(i, 1) * dt);
			double rate = B2.dot(X);
			rates_derivs.row(i) << rate, B2.transpose();

		}
	}
	return rates_derivs;

}



inline VectorXd fut_price_3m(const MatrixXd Data, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{



	Matrix3d Sigma;
	Sigma <<sigma11, sigma12, sigma13,
		sigma21, sigma22, sigma23,
		sigma31, sigma32, sigma33;

	Matrix3d Sigmasq = Sigma * Sigma.transpose();

	VectorXd rates(Data.rows());
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{
			double A_end = AFNS3_A(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, dt * Data(i, 3));
			VectorXd B_end = AFNS3_B2(lambda, dt * Data(i, 3));
			double rh = Data(i, 4) * exp(A_end - (B_end.dot(X)));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));


		}
		else
		{
			double A_end = AFNS3_A(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, dt * Data(i, 1));
			VectorXd B_end = AFNS3_B2(lambda, dt * Data(i, 1));

			double t = Data(i, 2) * dt; // Time to start

			double t1= (-exp( lambda * (-t)) + 1.0)/ ( lambda);  // e^(-k (t - x))
			double t2 = (-exp(2.0 * lambda * (-t)) + 1.0) / (2.0 * lambda); // e^(-2 k (t - x))
			double t3 = (exp(lambda * (-t)) * (lambda * (-t) - 1.0) + 1.0) / (lambda);// e^(-k (t - x)) k (t - x)
			double t4 = (exp(2.0*lambda * (-t)) * (2.0*lambda * (-t) - 1.0) + 1.0) / (4.0*lambda);// e^(-2 k (t - x)) k (t - x)
			double t5 = (1.0-exp(2.0 * lambda * (-t)) * ((2.0 * lambda * (t) )*(lambda*(t) + 1.0)+1)) / (4.0 * lambda*lambda);// e^(-2 k (t - x)) k (t - x)^2

			Matrix3d omega(3, 3);
			omega <<
				Sigmasq(0,0) * t,
				t1* Sigmasq(0, 1) + t3 * Sigmasq(0, 2),
				t1* Sigmasq(0, 2),
				t1* Sigmasq(1, 0) + t3 * Sigmasq(2, 0),
				t2* Sigmasq(1, 1) + t4 * Sigmasq(2, 1) + lambda * t4 * Sigmasq(1, 2) + lambda * t5 * Sigmasq(2, 2),
				t2* Sigmasq(1, 2) + t4 * Sigmasq(2, 2),
				t1* Sigmasq(2, 0),
				t2* Sigmasq(2, 1) + t4 * Sigmasq(2, 2),
				t2* Sigmasq(2, 2);
			

			Matrix3d analytic;
			analytic << 1.0, 0.0, 0.0,
				0.0, exp(-lambda * t), lambda* t* exp(-lambda * t),
				0., 0., exp(-lambda * t);

			double rh = exp(A_end) * exp(-(analytic * X).dot(B_end) + (0.5 * omega * B_end).dot(B_end));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));

		}
	}
	return rates;
}

inline MatrixXd fut_price_1m_sim(const VectorXd Starts, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{
	//VectorXd rates(Starts.rows());
	MatrixXd rates_derivs(Starts.rows(), 4);
	for (size_t i = 0; i < Starts.rows(); i++)
	{
		if (Starts(i) < 1)
		{

			Vector3d B_end = AFNS3_B2(lambda, dt * 30.0);
			Vector3d B = -B_end * (1.0 / (30.0 * dt));
			double rate =  B.dot(X);
			rates_derivs.row(i) << rate, B.transpose();




		}
		else
		{

			Vector3d B_start = AFNS3_B2(lambda, dt * Starts(i));
			Vector3d B_end = AFNS3_B2(lambda, dt * (Starts(i) + 30.0));
			Vector3d B = (B_start - B_end) / (30.0 * dt);
			double rate = B.dot(X);
			rates_derivs.row(i) << rate, B.transpose();


		}
	}
	return rates_derivs;

}


inline VectorXd fut_price_3m_sim(const VectorXd Starts, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{
	Matrix3d Sigma;
	Sigma << sigma11, sigma12, sigma13,
		sigma21, sigma22, sigma23,
		sigma31, sigma32, sigma33;

	Matrix3d Sigmasq = Sigma * Sigma.transpose();
	VectorXd rates(Starts.rows());
	for (size_t i = 0; i < Starts.rows(); i++)
	{
		if (Starts(i) < 1)
		{
			double A_end = AFNS3_A(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, dt * 90.0);
			VectorXd B_end = AFNS3_B2(lambda, dt * 90.0);
			double rh =  exp(A_end - (B_end.dot(X)));
			rates(i) = (rh - 1.0) / ((dt *90.0));
		}
		else
		{
			double A_end = AFNS3_A(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, dt * 90.0);
			VectorXd B_end = AFNS3_B2(lambda, dt * 90.0);

			double t = Starts(i) * dt; // Time to start

			double t1 = (-exp(lambda * (-t)) + 1.0) / (lambda);  // e^(-k (t - x))
			double t2 = (-exp(2.0 * lambda * (-t)) + 1.0) / (2.0 * lambda); // e^(-2 k (t - x))
			double t3 = (exp(lambda * (-t)) * (lambda * (-t) - 1.0) + 1.0) / (lambda);// e^(-k (t - x)) k (t - x)
			double t4 = (exp(2.0 * lambda * (-t)) * (2.0 * lambda * (-t) - 1.0) + 1.0) / (4.0 * lambda);// e^(-2 k (t - x)) k (t - x)
			double t5 = (1.0 - exp(2.0 * lambda * (-t)) * ((2.0 * lambda * (t)) * (lambda * (t)+1.0) + 1)) / (4.0 * lambda * lambda);// e^(-2 k (t - x)) k (t - x)^2


			Matrix3d omega(3, 3);
			omega <<
				Sigmasq(0, 0) * t,
				t1* Sigmasq(0, 1) + t3 * Sigmasq(0, 2),
				t1* Sigmasq(0, 2),
				t1* Sigmasq(1, 0) + t3 * Sigmasq(2, 0),
				t2* Sigmasq(1, 1) + t4 * Sigmasq(2, 1) + lambda * t4 * Sigmasq(1, 2) + lambda * t5 * Sigmasq(2, 2),
				t2* Sigmasq(1, 2) + t4 * Sigmasq(2, 2),
				t1* Sigmasq(2, 0),
				t2* Sigmasq(2, 1) + t4 * Sigmasq(2, 2),
				t2* Sigmasq(2, 2);

			Matrix3d analytic;
			analytic << 1.0, 0.0, 0.0,
				0.0, exp(-lambda * t), lambda* t* exp(-lambda * t),
				0., 0., exp(-lambda * t);

			double rh = exp(A_end) * exp(-(analytic * X).dot(B_end) + (0.5 * omega * B_end).dot(B_end));
			rates(i) = (rh - 1.0) / ((dt * 90.0));
		}
	}
	return rates;
}








inline double cond_vol(const double Start, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{
	Matrix3d Sigma;
	Sigma << sigma11, sigma12, sigma13,
		sigma21, sigma22, sigma23,
		sigma31, sigma32, sigma33;

	Matrix3d Sigmasq = Sigma * Sigma.transpose();

			double A_end = AFNS3_A(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, dt * 90.0);
			VectorXd B_end = AFNS3_B2(lambda, dt * 90.0);

			double t = Start * dt; // Time to start

			double t1 = (-exp(lambda * (-t)) + 1.0) / (lambda);  // e^(-k (t - x))
			double t2 = (-exp(2.0 * lambda * (-t)) + 1.0) / (2.0 * lambda); // e^(-2 k (t - x))
			double t3 = (exp(lambda * (-t)) * (lambda * (-t) - 1.0) + 1.0) / (lambda);// e^(-k (t - x)) k (t - x)
			double t4 = (exp(2.0 * lambda * (-t)) * (2.0 * lambda * (-t) - 1.0) + 1.0) / (4.0 * lambda);// e^(-2 k (t - x)) k (t - x)
			double t5 = (1.0 - exp(2.0 * lambda * (-t)) * ((2.0 * lambda * (t)) * (lambda * (t)+1.0) + 1)) / (4.0 * lambda * lambda);// e^(-2 k (t - x)) k (t - x)^2


			Matrix3d omega(3, 3);
			omega <<
				Sigmasq(0, 0) * t,
				t1* Sigmasq(0, 1) + t3 * Sigmasq(0, 2),
				t1* Sigmasq(0, 2),
				t1* Sigmasq(1, 0) + t3 * Sigmasq(2, 0),
				t2* Sigmasq(1, 1) + t4 * Sigmasq(2, 1) + lambda * t4 * Sigmasq(1, 2) + lambda * t5 * Sigmasq(2, 2),
				t2* Sigmasq(1, 2) + t4 * Sigmasq(2, 2),
				t1* Sigmasq(2, 0),
				t2* Sigmasq(2, 1) + t4 * Sigmasq(2, 2),
				t2* Sigmasq(2, 2);

			
			
				
	return sqrt((1.0/(0.25*0.25))*(omega * B_end).dot(B_end));
}








inline double AFNS2_A(const double sigma11, const double sigma22, const double lambda, const double time)
{

	const double e1 =(1.0/lambda)* (1.0-exp(-lambda * time));

	const double t1= sigma11*sigma11*time * time*time / 6.0;

	// Forkert ???
	const double t2 = time * sigma22 * sigma22 * (1.0 / (2.0 * lambda * lambda)) * (1.0 - (1.0 / time) * e1 - (1.0 / (2.0 * time))*lambda* e1 * e1);

	return t1+t2;
}


inline MatrixXd AFNS2_B(const double lambda, const double Maturity)
{
	VectorXd B(2);

	B(0) = -Maturity;
	B(1) = -1.0 / (lambda) * (1.0 - exp(-lambda * Maturity));

	return B;
}



inline MatrixXd fut_price_1m_2(const MatrixXd Data, const double dt, const double lambda, const VectorXd X)
{
	//VectorXd rates(Data.rows());
	MatrixXd rates_derivs(Data.rows(), 3);
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{

			Vector2d B_end = AFNS2_B(lambda, dt * Data(i, 3));
			Vector2d B = (-B_end) / (Data(i, 1) * dt);
			double rate = Data(i, 4) + B.dot(X);
			rates_derivs.row(i) << rate, B.transpose();


		}
		else
		{
			Vector2d B_start = AFNS2_B(lambda, dt * Data(i, 2));
			Vector2d B_end = AFNS2_B(lambda, dt * Data(i, 3));
			Vector2d B = (B_start - B_end) / (Data(i, 1) * dt);
			double rate = B.dot(X);
			rates_derivs.row(i) << rate, B.transpose();

		}
	}
	return rates_derivs;
}



inline VectorXd fut_price_3m_2(const MatrixXd Data, const double dt, const double sigma11, const double sigma22, const double lambda, const VectorXd X)
{
	VectorXd rates(Data.rows());
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{
			double A_end = AFNS2_A(sigma11,sigma22, lambda, dt * Data(i, 3));
			VectorXd B_end = AFNS2_B(lambda, dt * Data(i, 3));
			double rh = Data(i, 4) * exp(A_end - (B_end.dot(X)));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));
		}
		else
		{
			double A_end = AFNS2_A(sigma11, sigma22, lambda, dt * Data(i, 1));
			VectorXd B_end = AFNS2_B(lambda, dt * Data(i, 1));

			double t = Data(i, 2) * dt; // Time to start

			double first = (-exp(-2.0 * lambda * t) + 1.0) * 1.0 / (2.0 * lambda);
			double second = (-exp(-2.0 * lambda * t) * (2.0 * lambda * t + 1.0) + 1.0) / (4.0 * lambda);
			Matrix2d omega;
			omega << sigma11 * sigma11 * t, 0.0,
				0., sigma22* sigma22*(-exp(-2.0 * lambda * t)+1.0) / (2.0 * lambda);

			Matrix2d analytic;
			analytic << 1.0, 0.0, 
				0.0, exp(-lambda * t);

			double rh = exp(A_end) * exp(-(analytic * X).dot(B_end) + (0.5 * omega * B_end).dot(B_end));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));
		}
	}
	return rates;
}






inline double Vasicek_A(const double theta, const double sigma, const double lambda, const double time)
{

	const double e1 = (1.0 / lambda) * (1.0 - exp(-lambda * time));

	const double A = (theta - (sigma * sigma / (2.0 * lambda * lambda))) * (e1 - time) - ((sigma * sigma / (4 * lambda)) * e1 * e1);

	return A;
}


inline double Vasicek_B(const double lambda, const double Maturity)
{
	const double B = -1.0 / (lambda) * (1.0 - exp(-lambda * Maturity));
	return B;
}


inline double Vasicek_var(const double sigma, const double lambda, const double time)
{

	const double e1 = (1.0 / lambda) * (1.0 - exp(-lambda * time));

	const double var = (sigma * sigma / ( lambda * lambda)) * ((time-e1) - ((lambda / 2.0) * e1 * e1));

	return var;
}


inline double Vasicek_mean(const double theta, const double lambda, const double Maturity, const double X)
{
	const double B = -Vasicek_B(lambda, Maturity);
	const double mean = (X - theta) * B + theta * Maturity;
	return mean;
}






inline MatrixXd fut_price_1m_vas(const MatrixXd Data, const double dt, const double theta, const double lambda, const double X)
{
	//VectorXd rates(Data.rows());
	MatrixXd rates_derivs(Data.rows(), 2);
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{

			double B_end = Vasicek_B(lambda, dt * Data(i, 3));
			double B = (-B_end) / (Data(i, 1) * dt);
			double rate = Data(i, 4) + ((X-theta)*(-B_end)+theta* dt * Data(i, 3))/ (Data(i, 1) * dt);
			rates_derivs.row(i) << rate, B;

		}
		else
		{
			double B_start = Vasicek_B(lambda, dt * Data(i, 2));
			double B_end = Vasicek_B(lambda, dt * Data(i, 3));
			double B = (B_start - B_end) / (Data(i, 1) * dt);
			double rate = ((X - theta)*(B_start - B_end)+theta* Data(i, 1) * dt) / (Data(i, 1) * dt);
			rates_derivs.row(i) << rate, B;



		}
	}
	return rates_derivs;
}



inline VectorXd fut_price_3m_vas(const MatrixXd Data, const double dt, const double theta, const double sigma, const double lambda, const double X)
{
	VectorXd rates(Data.rows());
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{
			//double A_end = Vasicek_A(theta,sigma, lambda, dt * Data(i, 3));
			double B_end = Vasicek_B(lambda, dt * Data(i, 3));

			double var_end = Vasicek_var(sigma, lambda, dt * Data(i, 3));
			//double rh = Data(i, 4) * exp(A_end - (B_end*X));
			double rh = Data(i, 4) * exp(0.5*var_end + theta * B_end + theta * dt * Data(i, 3) - (B_end * X));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));
		}
		else
		{
			//double A_end = Vasicek_A(theta, sigma, lambda, dt * Data(i, 1));
			
			double B_end = Vasicek_B(lambda, dt * Data(i, 1));

			double t = Data(i, 2) * dt; // Time to start

			double omega = (sigma*sigma/(2.0*lambda))*(1.0-exp(-2.0*lambda*t));

			double analytic=X*exp(-lambda*t)+theta*(1.0-exp(-lambda *t)) ;

			//double rh = exp(A_end) * exp(-(analytic *B_end) + (0.5 * omega * B_end*B_end));

			double var_end = Vasicek_var(sigma, lambda, dt * Data(i, 1));

			double rh = exp(0.5*var_end+theta*B_end+theta* dt * Data(i, 1)) * exp(-(analytic * B_end) + (0.5 * omega * B_end * B_end));
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));
		}
	}
	return rates;
}























inline double forward_yield_adjustment(const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const double time)
{
	const double e1 = exp(-lambda * time);

	const double e2 = exp(-2.0 * lambda*time);

	const double l1 = -0.5*sigma11*sigma11 * time *time - 0.5*(sigma22 *sigma22 + sigma21 * sigma21)*pow((1.0 - e1) / lambda, 2.0);
	const double l2 = -0.5*(sigma31*sigma31 + sigma32 * sigma32 + sigma33 * sigma33)*(1.0 / (lambda*lambda) - 2.0 * e1 / (lambda*lambda) - 2.0 * time*e1 / lambda + e2 / (lambda *lambda) + 2.0 * time*e2 / lambda + (time*time)*e2);
	const double l3 = -(sigma11*sigma21*time*(1.0 - e1) / lambda) - sigma11 * sigma31*(time / lambda - time * e1 / lambda - (time*time)*e1);
	const double l4 = -(sigma21*sigma31 + sigma22 * sigma32)*(1.0 / (lambda*lambda) - 2.0 * e1 / (lambda*lambda) - time * e1 / lambda + e2 / (lambda*lambda) + time * e2 / lambda);

	return l1 + l2 + l3 + l4;
}

inline MatrixXd AFNS3_B(const double lambda, const VectorXd Maturities)
{
	MatrixXd B(Maturities.rows(), 3);
	for (size_t i = 0; i < Maturities.rows(); i++)
	{
		B(i, 0) = 1.0;
		B(i, 1) = (1.0 - exp(-lambda * Maturities(i))) / (lambda*Maturities(i));
		B(i, 2) = B(i, 1) - exp(-lambda * Maturities(i));
	}
	return B;
}




inline MatrixXd forward_B(const double lambda, const VectorXd time)
{
	MatrixXd B(time.rows(), 3);
	for (size_t i = 0; i < time.rows(); i++)
	{
		B(i, 0) = 1.0;
		B(i, 1) =  exp(-lambda * time(i));
		B(i, 2) = lambda * time(i)*B(i, 1);
	}
	return B;
}


inline double Squared_omega(const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const double time)
{
	const double e1 = exp(-lambda * time);

	const double e2 = exp(-2.0 * lambda*time);

	const double l1 = sigma11 * sigma11*time + (sigma21 *sigma21 + sigma22 * sigma22)*(1.0 - e2) / (2.0 * lambda);
	const double l2 = (sigma31*sigma31 + sigma32 * sigma32 + sigma33 * sigma33)*((1.0 - e2) / (4.0 * lambda) - 0.5*time*e2 - 0.5*lambda*time*time * e2);
	const double l3 = 2.0 * sigma11*sigma21*(1.0 - e1) / lambda + 2.0 * sigma11*sigma31*(-time * e1 + (1.0 - e1) / lambda);
	const double l4 = (sigma21*sigma31 + sigma22 * sigma32)*(-time * e2 + (1.0 - e2) / (2.0 * lambda));

	return l1 + l2 + l3 + l4;
}



inline VectorXd krip_forward(const Vector3d X, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd time)
{
	MatrixXd B = forward_B(lambda, time);
	VectorXd A(time.rows());
	for (size_t i = 0; i < time.rows(); i++)
	{
		A(i) = forward_yield_adjustment(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, time(i));
	}


	VectorXd forward=A+B*X ;


	VectorXd omega(time.rows());

	for (size_t i = 0; i < time.rows(); i++)
	{
		omega(i) =sqrt( Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, time(i)) );
	}

	VectorXd krip_forward(time.rows());

	for (size_t i = 0; i < time.rows(); i++)
	{
		krip_forward(i) = forward(i)*normalCdf(forward(i) / omega(i)) +omega(i)*normalDens(forward(i) / omega(i));
	}

	return krip_forward;

}

inline VectorXd zlb_yield(const Vector3d X, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda)
{

	VectorXd my_seq(1440);
	for (size_t i = 0; i < 1440; i++)
	{
		my_seq(i) = (i + 1.0) / 96.0;
	}

	VectorXd zlb_forwards = krip_forward(X, sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, my_seq);

	const double f1 = zlb_forwards.topRows(24).sum()*(1.0 / 96.0)/0.25;
	const double f2 = zlb_forwards.topRows(48).sum()*(1.0 / 96.0)/0.5;
	const double f3 = zlb_forwards.topRows(96).sum()*(1.0 / 96.0)/1.0;
	const double f4 = zlb_forwards.topRows(192).sum()*(1.0 / 96.0)/2.0;
	const double f5 = zlb_forwards.topRows(288).sum()*(1.0 / 96.0)/3.0;
	const double f6 = zlb_forwards.topRows(480).sum()*(1.0 / 96.0)/5.0;
	const double f7 = zlb_forwards.topRows(672).sum()*(1.0 / 96.0)/7.0;
	const double f8 = zlb_forwards.topRows(960).sum()*(1.0 / 96.0)/10.0;
	const double f9 = zlb_forwards.topRows(1440).sum()*(1.0 / 96.0)/15.0;

	VectorXd zlb_yields(9);
	zlb_yields << f1,f2,f3,f4,f5,f6,f7,f8,f9;

	return zlb_yields;

}


inline double g(const double Lambda,const double Maturity)
{
	return 1.0 / Lambda * (1.0 - exp(-Lambda * Maturity));
}


inline MatrixXd zlb_yields_jacobian(const Vector3d X, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd time, const double r_L)
{
	MatrixXd zlb_forwards_derivs(time.rows(), 4);
	double loading2, loading3, yield_adj, forward, omega;

	for (size_t i = 0; i < time.rows(); i++)
	{
		loading2 = exp(-lambda * time(i));
		loading3 = lambda * time(i)*loading2;
		yield_adj = forward_yield_adjustment(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, time(i));
		forward = X(0) + X(1)*loading2 + X(2)*loading3 + yield_adj;

		omega = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, time(i)));

		zlb_forwards_derivs(i,1)= normalCdf((forward - r_L) / omega);

		zlb_forwards_derivs(i, 2) = loading2 * zlb_forwards_derivs(i, 1);

		zlb_forwards_derivs(i, 3) = loading3 * zlb_forwards_derivs(i, 1);
		
		zlb_forwards_derivs(i, 0) =  r_L + (forward - r_L)*zlb_forwards_derivs(i, 1) + omega * normalDens((forward - r_L) / omega);

	}

	MatrixXd zlb_yields_jacobian(9, 4);

	for (size_t j = 0; j < 4; j++)
	{
		zlb_yields_jacobian(0, j) = zlb_forwards_derivs.block<24,1>(0,j).mean();
		zlb_yields_jacobian(1, j) = zlb_forwards_derivs.block<48, 1>(0, j).mean();
		zlb_yields_jacobian(2, j) = zlb_forwards_derivs.block<96, 1>(0, j).mean();
		zlb_yields_jacobian(3, j) = zlb_forwards_derivs.block<192, 1>(0, j).mean();
		zlb_yields_jacobian(4, j) = zlb_forwards_derivs.block<288, 1>(0, j).mean();
		zlb_yields_jacobian(5, j) = zlb_forwards_derivs.block<480, 1>(0, j).mean();
		zlb_yields_jacobian(6, j) = zlb_forwards_derivs.block<672, 1>(0, j).mean();
		zlb_yields_jacobian(7, j) = zlb_forwards_derivs.block<960, 1>(0, j).mean();
		zlb_yields_jacobian(8, j) = zlb_forwards_derivs.block<1440, 1>(0, j).mean();
	}

	return zlb_yields_jacobian;

}




inline VectorXd fut_price_1m_shadow(const MatrixXd Data, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{

	//Vector3d rho1(1., 1., 0.0);
	//MatrixXd rates_derivs(Data.rows(), 4);
	VectorXd rates(Data.rows());

	auto m1 = [&](double s) {

		double vol = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));
		
		double mean = X(0) + exp(-lambda * s) * X(1) +  lambda * s * exp(-lambda * s) * X(2);

		return mean * normalCdf(mean / vol) + vol * normalDens(mean / vol);
	};
	/*double test;
	for (int h = 0; h < Data(0, 3); h++)
	{
		test += dt * m1(h * dt);
	}

	cout << test << " eller  " << gauss_kronrod<double, 15>::integrate(m1, 0.0, dt * Data(0, 3), 0,0);*/

	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{
			rates(i) = Data(i, 4) + (1.0/ (Data(i, 1)*dt))*gauss_kronrod<double, 15>::integrate(m1, 0.0, dt * Data(i, 3), 0, 0);
		}
		else
		{
			rates(i) =  1.0/(Data(i, 1)*dt)*gauss_kronrod<double, 15>::integrate(m1, dt * Data(i, 2), dt * Data(i, 3), 0, 0);
		}
	}
	return rates;
}


inline VectorXd fut_price_3m_shadow(const MatrixXd Data, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{


	auto m1 = [&](double s) {

		double vol = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));

		double mean = X(0) + exp(-lambda * s) * X(1) + lambda * s * exp(-lambda * s) * X(2);

		return mean * normalCdf(mean / vol) + vol * normalDens(mean / vol);
	};




	VectorXd rates(Data.rows());
	for (size_t i = 0; i < Data.rows(); i++)
	{
		if (Data(i, 2) < 1)
		{


			double cum1 = gauss_kronrod<double, 15>::integrate(m1, 0.0, dt * Data(i, 3), 0, 0);


			auto m2 = [&](double u, double s) {
				double vol_u = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, u));
				double mean_u = X(0) + exp(-lambda * u) * X(1) + (exp(-lambda * u) + lambda * u * exp(-lambda * u)) * X(2);

				double vol_s = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));
				double mean_s = X(0) + exp(-lambda * s) * X(1) + (exp(-lambda * s) + lambda * s * exp(-lambda * s)) * X(2);

				double zeta_u = mean_u / vol_u;
				double zeta_s = mean_s / vol_s;

				double Cov = sigma11 * sigma11 * min(u, s) + sigma22 * sigma22 / (2.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) - exp(-lambda * (s + u)))
					+ sigma33 * sigma33 / (4.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) * (lambda * (s + u - 2.0 * min(u, s)) + 1.0) - exp(-lambda * (s + u)) * (2.0 * lambda * lambda * s * u + lambda * (s + u) + 1.0));


				double Chi = Cov / (vol_s * vol_u);

				double decumulative = bivnor(-zeta_u, -zeta_s, Chi);

				return ((mean_u * mean_s + Cov) * decumulative
					+ vol_s * mean_u * normalDens(zeta_s) * normalCdf((zeta_u - Chi * zeta_s) / sqrt(1.0 - Chi * Chi))
					+ vol_u * mean_s * normalDens(zeta_u) * normalCdf((zeta_s - Chi * zeta_u) / sqrt(1.0 - Chi * Chi))
					+ vol_u * vol_s * sqrt(1.0 - Chi * Chi) / 2.506628274631 * normalDens(sqrt((zeta_u * zeta_u - 2.0 * Chi * zeta_u * zeta_s + zeta_s * zeta_s) / (1.0 - Chi * Chi))));
			};
			int p = 1;
			auto f = [&](double s) {
				auto g = [&](double u) {
					return m2(u, s);
				};
				return gauss_kronrod<double, 5>::integrate(g, 0, s, 0, 0);
			};



			double secondmoment = 2.0*gauss_kronrod<double, 5>::integrate(f, 0.0, dt * Data(i, 3) , 0, 0);

			double cum2 = secondmoment - cum1 * cum1;

			
			double rh = Data(i, 4) * exp(cum1+0.5*cum2);
			rates(i) = (rh - 1.0) / ((dt * Data(i, 1)));


		}

		else
		{	
			double t_start = Data(i, 2) * dt; // Time to start

			double cum1 = gauss_kronrod<double, 15>::integrate(m1, t_start , dt * Data(i, 3), 0, 0);


			auto m2 = [&](double u, double s) {
				double vol_u = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, u));
				double mean_u = X(0) + exp(-lambda * u) * X(1) + (exp(-lambda * u) + lambda * u * exp(-lambda * u)) * X(2);

				double vol_s = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));
				double mean_s = X(0) + exp(-lambda * s) * X(1) + (exp(-lambda * s) + lambda * s * exp(-lambda * s)) * X(2);

				double zeta_u = mean_u / vol_u;
				double zeta_s = mean_s / vol_s;


				double Cov = sigma11 * sigma11 * min(u, s) + sigma22 * sigma22 / (2.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) - exp(-lambda * (s + u)))
					+ sigma33 * sigma33 / (4.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) * (lambda * (s + u - 2.0 * min(u, s)) + 1.0) - exp(-lambda * (s + u)) * (2.0 * lambda * lambda * s * u + lambda * (s + u) + 1.0));

				double Chi =  Cov / (vol_s * vol_u);


				double decumulative = bivnor(-zeta_u, -zeta_s, Chi);

				return ((mean_u * mean_s + Cov) * decumulative
					+ vol_s * mean_u * normalDens(zeta_s) * normalCdf((zeta_u - Chi * zeta_s) / sqrt(1.0 - Chi * Chi))
					+ vol_u * mean_s * normalDens(zeta_u) * normalCdf((zeta_s - Chi * zeta_u) / sqrt(1.0 - Chi * Chi))
					+ vol_u * vol_s * sqrt(1.0 - Chi * Chi) / 2.506628274631 * normalDens(sqrt((zeta_u * zeta_u - 2.0 * Chi * zeta_u * zeta_s + zeta_s * zeta_s) / (1.0 - Chi * Chi))));
			};

			auto f = [&](double s) {
				auto g = [&](double u) {
					return m2(u, s);
				};
				return gauss_kronrod<double, 5>::integrate(g, t_start, s, 0, 0);
			};

			double secondmoment = 2.0*gauss_kronrod<double, 5>::integrate(f, t_start, dt * Data(i, 3), 0, 0);

			double cum2 = secondmoment - cum1 * cum1;

			rates(i) = (exp(cum1+0.5*cum2) - 1.0) / ((dt * Data(i, 1)));

		}
	}
	return rates;
}












inline double Shadow_bond(const double Mat, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{


	auto m1 = [&](double s) {

		double vol = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));

		double mean = X(0) + exp(-lambda * s) * X(1) + lambda * s * exp(-lambda * s) * X(2);

		return mean * normalCdf(mean / vol) + vol * normalDens(mean / vol);
	};

	double cum1 = gauss_kronrod<double, 61>::integrate(m1, 0.0, Mat, 0, 0);


			auto m2 = [&](double u, double s) {
				double vol_u = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, u));
				double mean_u = X(0) + exp(-lambda * u) * X(1) + (exp(-lambda * u) + lambda * u * exp(-lambda * u)) * X(2);

				double vol_s = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));
				double mean_s = X(0) + exp(-lambda * s) * X(1) + (exp(-lambda * s) + lambda * s * exp(-lambda * s)) * X(2);

				double zeta_u = mean_u / vol_u;
				double zeta_s = mean_s / vol_s;

				

				double Cov = sigma11 * sigma11 * min(u, s) + sigma22 * sigma22 / (2.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) - exp(-lambda * (s + u)))
					+ sigma33 * sigma33 / (4.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) * (lambda * (s + u - 2.0 * min(u, s)) + 1.0) - exp(-lambda * (s + u)) * (2.0 * lambda * lambda * s * u + lambda * (s + u) + 1.0));
				
				double Chi = Cov / (vol_s * vol_u);

				double decumulative =  bivnor(-zeta_u, -zeta_s, Chi);

				return ((mean_u * mean_s + Cov) * decumulative
					+ vol_s * mean_u * normalDens(zeta_s) * normalCdf((zeta_u - Chi * zeta_s) / sqrt(1.0 - Chi * Chi))
					+ vol_u * mean_s * normalDens(zeta_u) * normalCdf((zeta_s - Chi * zeta_u) / sqrt(1.0 - Chi * Chi))
					+ vol_u * vol_s * sqrt(1.0 - Chi * Chi) / 2.506628274631 * normalDens(sqrt((zeta_u * zeta_u - 2.0 * Chi * zeta_u * zeta_s + zeta_s * zeta_s) / (1.0 - Chi * Chi))));
			};

			int k = 0;
			auto f = [&](double s) {
				auto g = [&](double u) {
					return m2(u, s);
				};
				return gauss_kronrod<double, 31>::integrate(g, 0.0, s, 0, 0);
			};



			double secondmoment = 2.0*gauss_kronrod<double, 31>::integrate(f, 0.0, Mat, 0, 0);

			double cum2 = secondmoment - cum1 * cum1;
		
	return exp(-cum1 + 0.5 * cum2);
}





inline double fut_1m_shadow(const double start, const double end, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{
	auto m1 = [&](double s) {

		double vol = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));

		double mean = X(0) + exp(-lambda * s) * X(1) + lambda * s * exp(-lambda * s) * X(2);

		return mean * normalCdf(mean / vol) + vol * normalDens(mean / vol);
	};
		
	double rates = 1.0 / (1.0/12.0) * gauss_kronrod<double, 15>::integrate(m1, start, end, 0, 0);
	
	return rates;
}


inline double fut_3m_shadow(const double start, const double end, const double dt, const double sigma11, const double sigma12, const double sigma13, const double sigma21, const double sigma22, const double sigma23, const double sigma31, const double sigma32, const double sigma33, const double lambda, const VectorXd X)
{


	auto m1 = [&](double s) {

		double vol = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));

		double mean = X(0) + exp(-lambda * s) * X(1) + lambda * s * exp(-lambda * s) * X(2);

		return mean * normalCdf(mean / vol) + vol * normalDens(mean / vol);
	};



	double cum1 = gauss_kronrod<double, 15>::integrate(m1, start, end, 0, 0);


			auto m2 = [&](double u, double s) {
				double vol_u = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, u));
				double mean_u = X(0) + exp(-lambda * u) * X(1) + (exp(-lambda * u) + lambda * u * exp(-lambda * u)) * X(2);

				double vol_s = sqrt(Squared_omega(sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33, lambda, s));
				double mean_s = X(0) + exp(-lambda * s) * X(1) + (exp(-lambda * s) + lambda * s * exp(-lambda * s)) * X(2);

				double zeta_u = mean_u / vol_u;
				double zeta_s = mean_s / vol_s;



				double Cov = sigma11 * sigma11 * min(u, s) + sigma22 * sigma22 / (2.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) - exp(-lambda * (s + u)))
					+ sigma33 * sigma33 / (4.0 * lambda) * (exp(-lambda * (s + u - 2.0 * min(u, s))) * (lambda * (s + u - 2.0 * min(u, s)) + 1.0) - exp(-lambda * (s + u)) * (2.0 * lambda * lambda * s * u + lambda * (s + u) + 1.0));


				double Chi = Cov / (vol_s * vol_u);


				double decumulative = bivnor(-zeta_u, -zeta_s, Chi);

				return ((mean_u * mean_s + Cov) * decumulative
					+ vol_s * mean_u * normalDens(zeta_s) * normalCdf((zeta_u - Chi * zeta_s) / sqrt(1.0 - Chi * Chi))
					+ vol_u * mean_s * normalDens(zeta_u) * normalCdf((zeta_s - Chi * zeta_u) / sqrt(1.0 - Chi * Chi))
					+ vol_u * vol_s * sqrt(1.0 - Chi * Chi) / 2.506628274631 * normalDens(sqrt((zeta_u * zeta_u - 2.0 * Chi * zeta_u * zeta_s + zeta_s * zeta_s) / (1.0 - Chi * Chi))));
			};

			auto f = [&](double s) {
				auto g = [&](double u) {
					return m2(u, s);
				};
				return gauss_kronrod<double, 5>::integrate(g, start, s, 0, 0);
			};

			double secondmoment = 2.0*gauss_kronrod<double, 5>::integrate(f, start, end, 0, 0);

			double cum2 = secondmoment - cum1 * cum1;

			
	return (exp(cum1 + 0.5 * cum2) - 1.0) / (1.0 / 4.0);
}
