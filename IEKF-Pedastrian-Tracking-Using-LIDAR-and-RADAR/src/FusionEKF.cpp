#include "FusionEKF.h"
#include <iostream>
using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;
/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing measurement matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
               0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //Initialize process model
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;
  //Initialize state covariance matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1000, 0,
           0, 0, 0, 1000;
  //Initialize process model noise
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.noise_ax=9.0;
  ekf_.noise_ay=9.0;
  //State transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  
   //Identity matrix
  ekf_.I_ = MatrixXd::Identity(ekf_.x_.size(), ekf_.x_.size());

 
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      /*
        Polar To Cartesian:
        x=rho*cos(phi)
        y=rho*sin(phi)
        vx=rho_dot*cos(phi)-rho*sin(phi)*phi_dot  //assume phi_dot=0
        vy=rho_dot*sin(phi)+rho*cos(phi)*phi_dot  //assume phi_dot=0
      */

      float rho    = measurement_pack.raw_measurements_[0];
      float phi    = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];

      ekf_.x_ << rho*cos(phi), 
                 rho*sin(phi),
                 rho_dot*cos(phi), 
                 rho_dot*sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
                 measurement_pack.raw_measurements_[1], 
                 0, 
                 0;
    }
    previous_timestamp_ = measurement_pack.timestamp_ ;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
   double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
   previous_timestamp_ = measurement_pack.timestamp_;

    // State transition matrix update
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
  

  // Noise covariance matrix computation

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  
  ekf_.Q_ <<  dt_4/4*ekf_.noise_ax, 0, dt_3/2*ekf_.noise_ax, 0,
         0, dt_4/4*ekf_.noise_ay, 0, dt_3/2*ekf_.noise_ay,
         dt_3/2*ekf_.noise_ax, 0, dt_2*ekf_.noise_ax, 0,
         0, dt_3/2*ekf_.noise_ay, 0, dt_2*ekf_.noise_ay;

  ekf_.Predict();
  /*****************************************************************************
   *  Update
   ****************************************************************************/


  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    //ekf_.UpdateIEKF(measurement_pack.raw_measurements_); 

  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.UpdateKF(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}