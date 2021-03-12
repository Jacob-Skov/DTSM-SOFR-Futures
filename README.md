# Dynamic Term Structure Models for SOFR Futures

This reposity contains the key code parts to the paper "Dynamic Term Structure Models for SOFR Futures".
Specifically "SOFR_AFNS3diag.h" and "SOFR_shadow3diag.h" implements the Extended Kalman Filter used in the estimation of the three-facotr AFNS and shadow rate model.
Furthermore, "Auxiliary.h" contains the necessary functions to price one- and three-month SOFR futures contracts in the Gaussian models considered in the paper as well as the shadow rate extension. "Gaussians.h" contains results on standard the Gaussian distribution, while "Toms462.h" and ""Toms462.cpp" is used to compute the decumulative bivariate cumulative normal distribution function.

Note: The functions are created to fit a certain futures input data used in the article. Appropriate modifications therefore have to be made for it to run on different data.

The c++ code also has a number of dependencies which have to be included to run the code.

The following Libraries have been used:

Eigen (https://eigen.tuxfamily.org/index.php?title=Main_Page) \
Boost (https://www.boost.org/) \
Optimlib (https://github.com/PatWie/CppNumericalSolvers)

The code should be seen as inspiration to implement similar models as those presented in the paper for e.g. SOFR or EFFR futures. 
