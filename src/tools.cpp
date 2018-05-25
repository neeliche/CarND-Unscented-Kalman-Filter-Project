#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
        const vector<VectorXd> &ground_truth) {
    /**
    TODO:
     * Calculate the RMSE here.
     */
    cout<<"Inside RMSE";
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    // ... your code here
    if (estimations.size() != ground_truth.size()
            || estimations.size() == 0) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {
        
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = (residual.array() * residual.array());
        rmse += residual;
    }

    //calculate the mean
    // ... your code here
    rmse = rmse / estimations.size();

    //calculate the squared root
    // ... your code here
    rmse = sqrt(rmse.array());

    //return the result
    return rmse;
}