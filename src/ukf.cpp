#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    //cout << "Going to initialize" << endl;
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.3;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

    //Setting initialization to false
    is_initialized_ = false;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Sigma point spreading parameter
    lambda_ = 3 - n_x_;

    // Number of Sigma points
    n_sigma_ = 2 * n_aug_ + 1;

    // Weights of sigma points
    weights_ = VectorXd(n_sigma_);

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

    // Noise matrices
    R_radar = MatrixXd(3, 3);
    R_radar << std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0, std_radrd_*std_radrd_;

    R_laser = MatrixXd(2, 2);
    R_laser << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    //Initialize NIS variables
    NIS_laser_ = 0.0;
    NIS_radar_ = 0.0;

    //Initialize time. 
    time_us_ = 0.0;
    //cout << "Initialized" << endl;
}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    //cout << "Inside Process Measurement" << endl;

    if ((MeasurementPackage::RADAR == meas_package.sensor_type_ && use_radar_) ||
            (MeasurementPackage::LASER == meas_package.sensor_type_ && use_laser_)) {


        if (!is_initialized_) {

            //cout << " Not initialized" << endl;

            if (MeasurementPackage::RADAR == meas_package.sensor_type_) {
                //For RADAR values, we need to convert them from polar to Cartesian
                //coordinates
                //range
                double rho = meas_package.raw_measurements_(0);
                //bearing
                double phi = meas_package.raw_measurements_(1);
                //Velocity
                double rhodot = meas_package.raw_measurements_(2);

                double px = rho * cos(phi);
                double py = rho * sin(phi);
                double vx = rhodot * cos(phi);
                double vy = rhodot * sin(phi);
                double v = sqrt(vx * vx + vy * vy); //Apparently we can tune this value

                x_ << px, py, v, 0, 0;

                //cout << "radar" << x_ << endl;

            } else if (MeasurementPackage::LASER == meas_package.sensor_type_) {
                //Other option is gonna be LASER, if the data is clean. But....

                /**
                 * For LASER, we only have x and y measurement at start. We cannot
                 * calculate the velocity from that.
                 */
                x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
                //cout << "LIDAR" << endl << x_ << endl;
            }

            time_us_ = meas_package.timestamp_;
            is_initialized_ = true;
            return;
        }

        //Calculate the time difference
        double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
        time_us_ = meas_package.timestamp_;

        //cout << "Calling Prediction" << endl;
        //Call to predict the value
        Prediction(dt);

        //cout << "Calling Update" << endl;
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            //cout << "Radar Update" << endl;
            UpdateRadar(meas_package);
        }
        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            //cout << "Laser Update" << endl;
            UpdateLidar(meas_package);
        }
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
     */

    //Generating the Sigma points
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    //Calculate the square root of P
    MatrixXd A_ = P_.llt().matrixL();

    //set lambda for non-augmented sigma points
    lambda_ = 3 - n_x_;

    //cout << "Going to set Xsig" << endl;
    //cout << "Delta T" << delta_t << endl;
    Xsig.col(0) = x_;
    //set sigma points as columns of matrix Xsig
    for (int i = 0; i < n_x_; i++) {
        VectorXd sqrt_part = sqrt(lambda_ + n_x_) * A_.col(i);
        Xsig.col(i + 1) = x_ + sqrt_part;
        Xsig.col(i + 1 + n_x_) = x_ - sqrt_part;
    }
    //cout << Xsig << endl;


    //Augmenting Sigma points

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

    lambda_ = 3 - n_aug_;

    //cout << "Augmented Mean state" << endl;
    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //cout << "Augmented covariance matrix" << endl;
    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_*std_a_;
    P_aug(6, 6) = std_yawdd_*std_yawdd_;

    //cout << "square root matrix" << endl;
    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //cout << "Create Xsig_aug" << endl;
    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        VectorXd sq_part = sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1) = x_aug + sq_part;
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sq_part;
    }
    //cout << Xsig_aug << endl;


    //predict sigma points
    //cout << "Predict sigma points" << endl;
    for (int i = 0; i < n_sigma_; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
    //cout << "Finished sigma point calc" << endl;

    //cout << Xsig_pred_ << endl;

    // Weights of sigma points
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 0; i < 2 * n_aug_; i++) weights_(i + 1) = 1 / (2 * (lambda_ + n_aug_));
    //Finding the predicted mean and covariance

    //cout << "Predict State Mean" << endl;
    //predict state mean
    x_.fill(0.0); //This was the culprit behind me wasting 2 days of debugging.
    for (int i = 0; i < n_sigma_; i++) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //cout << "Predict state covariance" << endl;
    //predict state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3)<-M_PI) x_diff(3) += 2. * M_PI;

        P_ += weights_(i) * x_diff * x_diff.transpose();
    }

    //cout << "Prediction over" << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
     */

    VectorXd z = meas_package.raw_measurements_;
    //Predicting the measurement
    int n_z = 2;
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
        
        //mean predicted measurement
        z_pred += weights_(i) * Zsig.col(i);
    }

    UpdateUKF(meas_package, Zsig, z_pred, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
     */

    VectorXd z = meas_package.raw_measurements_;
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++) { //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); //r
        Zsig(1, i) = atan2(p_y, p_x); //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot

        //mean predicted measurement
        z_pred += weights_(i) * Zsig.col(i);
    }

    UpdateUKF(meas_package, Zsig, z_pred, n_z);
}

/**
 * A common function to update the Unsented Kalman Filter for both Radar
 * and Lidar measurements.
 * @param meas_package The measurement at k+1
 * @param Zsig The Z sigma points
 * @param z_pred The predicted Z
 * @param n_z The size of the Z Matrix (varies between Lidar and Radar)
 */
void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, VectorXd z_pred, int n_z) {

    //cout << "In UKF" << endl;
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);

    //calculate innovation covariance matrix S
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1)<-M_PI) z_diff(1) += 2. * M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }


    if (MeasurementPackage::RADAR == meas_package.sensor_type_ && use_radar_) {
        S += R_radar;
    }
    if (MeasurementPackage::LASER == meas_package.sensor_type_ && use_laser_) {
        S += R_laser;
    }


    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {


        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) <-M_PI) z_diff(1) += 2. * M_PI;


        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) <-M_PI) x_diff(3) += 2. * M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();


    }


    //extract measurement as VectorXd
    VectorXd z = meas_package.raw_measurements_;

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2. * M_PI;


    //calculate NIS
    if (MeasurementPackage::RADAR == meas_package.sensor_type_ && use_radar_) {
        NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    }
    if (MeasurementPackage::LASER == meas_package.sensor_type_ && use_laser_) {
        NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
    }


    //update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();


}


