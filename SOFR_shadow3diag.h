#pragma once

#include <math.h>

#include <fstream>
#include <iostream>
#include "Auxiliary.h"

#include "../../include/cppoptlib/meta.h"
#include "../../include/cppoptlib/problem.h"

#include <Eigen/Dense>
//#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>

using namespace cppoptlib;
using Eigen::VectorXd;

using namespace Eigen;
using namespace std;

class SOFR_shadow3diag : public Problem<double>
{
public:

	SOFR_shadow3diag(const MatrixXd& Data, const double& dt1, const double& dt2) :
		Data(Data), dt1(dt1), dt2(dt2)
	{}

	Vector3d get_Final_state()
	{
		return Final_state;
	}

	Vector3d get_Final_var()
	{
		return Final_var;
	}


	double value(const VectorXd& x)
	{
		const double dx = 0.00001;
		VectorXd hundreds(12);
		hundreds << 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0;

		Matrix3d K;
		K << x(0), 0.0, 0.0,
			0., x(1), 0,
			0., 0., x(2);


		VectorXd se = VectorXd::Zero(12);

		VectorXd vec_joined(Data.rows());

		EigenSolver<Matrix3d> myeigen(K);

		Vector3cd eigval = myeigen.eigenvalues();
		Matrix3cd eigvec = myeigen.eigenvectors();


		const Vector3d Theta(x(3), x(4), x(5));

		Matrix3d Sigma;
		Sigma << x(6), 0.0, 0.0,
			0., x(7), 0.0,
			0., 0., x(8);

		const double Lambda = x(9);

		VectorXd Varsvec(12);
		Varsvec << x(10) * x(10), x(11)* x(11), x(12)* x(12), x(13)* x(13), x(14)* x(14), x(15)* x(15), x(16)* x(16), x(17)* x(17), x(18)* x(18), x(19)* x(19), x(20)* x(20), x(21)* x(21);
		const MatrixXd Vars = Varsvec.asDiagonal();

		double loglike;
		double cum_loglike = 0.0;


		if (Lambda < 0 || eigval.real()(0) <= 0.0 || eigval.real()(1) <= 0.0 || eigval.real()(2) <= 0.0) { return 70000000.0; }
		else
		{
			Matrix3cd inveigenv = eigvec.inverse();

			Matrix3cd sigmasq = inveigenv * Sigma * Sigma.transpose() * inveigenv.transpose();



			Matrix3cd batman(3, 3);
			Matrix3cd VP(3, 3);

			for (size_t i = 0; i < 3; i++)
			{
				for (size_t j = 0; j < 3; j++)
				{
					batman(i, j) = sigmasq(i, j) / (eigval(i) + eigval(j)) * (1.0 - exp(-(eigval(i) + eigval(j)) * dt1));
					VP(i, j) = sigmasq(i, j) / (eigval(i) + eigval(j));
				}
			}



			Matrix3d varPost = (eigvec * VP * eigvec.transpose()).real();


			Matrix3d Q = (eigvec * batman * eigvec.transpose()).real();


			Matrix3d varPrior;

			Vector3d xPost1 = Theta;
			Vector3d xPrior, xPost2;
			VectorXd residuals, rates(12);
			MatrixXd Minv, Kgain, B(12, 3), B_1m(7, 3), B_3m(5, 3);


			double det_M;

			Matrix3d Kdt = -dt1 * K;

			//Matrix3d Phi1 = Kdt.exp();

			Matrix3d Phi1 = matrix_exponential(Kdt, 100);

			Vector3d Phi0 = (Matrix3d::Identity() - Phi1) * Theta;

			for (size_t i = 0; i < Data.rows(); i += 12)
			{
				MatrixXd quotes_1m = Data.block<7, 7>(i, 0);
				MatrixXd quotes_3m = Data.block<5, 7>(i + 7, 0);
				MatrixXd quotes = Data.block<12, 7>(i, 0);

				xPrior = Phi0 + Phi1 * xPost1;

				xPost1 = xPrior;

				varPrior = Phi1 * varPost * Phi1.transpose() + Q;


				int itr = 0;
				do
				{
					itr += 1;
					xPost2 = xPost1;


					VectorXd rates_1m = fut_price_1m_shadow(quotes_1m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2);

					VectorXd B1_1m = (fut_price_1m_shadow(quotes_1m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(dx, 0.0, 0.0)) - rates_1m) / dx;
					VectorXd B2_1m = (fut_price_1m_shadow(quotes_1m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(0.0, dx, 0.0)) - rates_1m) / dx;
					VectorXd B3_1m = (fut_price_1m_shadow(quotes_1m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(0.0, 0.0, dx)) - rates_1m) / dx;

					B_1m << B1_1m, B2_1m, B3_1m;


					VectorXd rates_3m = fut_price_3m_shadow(quotes_3m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2);

					VectorXd B1_3m = (fut_price_3m_shadow(quotes_3m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(dx, 0.0, 0.0)) - rates_3m) / dx;
					VectorXd B2_3m = (fut_price_3m_shadow(quotes_3m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(0.0, dx, 0.0)) - rates_3m) / dx;
					VectorXd B3_3m = (fut_price_3m_shadow(quotes_3m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost2 + Vector3d(0.0, 0.0, dx)) - rates_3m) / dx;

					B_3m << B1_3m, B2_3m, B3_3m;

					B << B_1m, B_3m;

					rates << rates_1m, rates_3m;


					residuals = (hundreds - quotes.col(0)) / 100.0 - rates - B * (xPrior - xPost2);


					MatrixXd M = B * varPrior * B.transpose() + Vars;

					det_M = M.determinant();

					if (isnan(det_M) || isinf(det_M)) { return 90000000.0; }
					else
					{
						if (det_M > 1e-150)
						{
							Minv = M.inverse();


							Kgain = varPrior * B.transpose() * Minv;

							xPost1 = xPrior + Kgain * residuals;

						}
						else
						{
							return 80000000.0;

						}

					}



					
				} while ((xPost1 - xPost2).cwiseAbs().maxCoeff() > 1e-7 && itr < 1); /* outcomment <1 to get iekf!*/

				varPost = (Matrix3d::Identity() - Kgain * B) * varPrior;

				loglike = 0.5 * (-12.0 * log(2.0 * 3.14159265358979323846) - log(det_M) - residuals.dot(Minv * residuals));

				//VectorXd rates_1 = fut_price_1m_shadow(quotes_1m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost1).leftCols(1);

				//VectorXd rates_3 = fut_price_3m_shadow(quotes_3m, dt2, Sigma(0, 0), Sigma(0, 1), Sigma(0, 2), Sigma(1, 0), Sigma(1, 1), Sigma(1, 2), Sigma(2, 0), Sigma(2, 1), Sigma(2, 2), Lambda, xPost1);

				//VectorXd vec_joined(12);
				//vec_joined << rates_1, rates_3;
				//VectorXd error = (hundreds - quotes.col(0)) / 100.0 - vec_joined;
				//se = se+error.cwiseProduct(error);

				cum_loglike += loglike;
			}


			//cout << (se / 378.0).cwiseSqrt() << endl;

			//ofstream myfile;
			//myfile.open("C:\\Users\\bvg170\\Desktop\\phd\\afns3d_fit.txt");
			//myfile << vec_joined;
			//myfile.close();

			Final_state = xPost1;
			Final_var = varPost.diagonal();
		}


		//std::cout << -cum_loglike << std::endl;
		return -cum_loglike;
	}







private:
	Vector3d Final_state;
	Vector3d Final_var;
	MatrixXd Data;
	double dt1;
	double dt2;
};
