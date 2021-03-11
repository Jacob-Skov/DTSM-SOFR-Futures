# DTSM-SOFR-Futures
Dynamic Term Structure Models for SOFR Futures

This reposity contains the key code parts to the paper "Dynamic Term Structure Models for SOFR Futures".
Specifically XX implements the Extended Kalman Filter used in the estimation of the three-facotr AFNS and shadow rate model.
Furthermore, XX contains the necessary functions to price one- and three-month SOFR futures contracts in the Gaussian models considered in the paper as well as the shadow rate extension.

Note: The functions are created to fit a certain futures input data used in the article. Appropriate modifications therefore have to be made for it to run on different data.

The c++ code also has a number of dependencies which have to be included to run the code.

The following Libraries have been used:
Eigen (https://eigen.tuxfamily.org/index.php?title=Main_Page)
Boost (https://www.boost.org/)
Numerical (https://github.com/PatWie/CppNumericalSolvers)

The code should be seen as inspiration to implement similar models for e.g. SOFR or EFFR futures. 