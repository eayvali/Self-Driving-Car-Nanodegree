#include "kalman_filter.h"
#include <math.h>
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in ,
                        MatrixXd &I_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;    
  I_ = I_in;


}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateKF(const VectorXd &z) {
  //z: [x,y,vx,vy]
  VectorXd h = H_ * x_;
  VectorXd y =  z -  h;
  UpdateState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //z:[rho.phi,rho_dot]
  //H=Hj(px,py,vx,vy)

  VectorXd h = VectorXd(3);
  h(0) = sqrt( x_(0)*x_(0)+ x_(1)*x_(1));

  if (fabs(h(0)) < 0.0001) {
    std::cout << "Division by Zero..Skipping radar measurement update" << std::endl;
    return;
  }

  h(1) = atan2( x_(1) , x_(0) );
  h(2) = (x_(0) * x_(2) + x_(1) * x_(3)) / h(0);

  VectorXd y =  z -  h;
  
  if (y(1) > M_PI){ 
    y(1) = y(1) - 2*M_PI;
  }
  else if (y(1) < -M_PI) {
    y(1) = y(1) + 2*M_PI;
  }
  UpdateState(y);

}

void KalmanFilter::UpdateState(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}


void KalmanFilter::UpdateIEKF(const VectorXd &z) {
  //z:[rho.phi,rho_dot]
  //H=Hj(px,py,vx,vy)
  int maxIterations=15;
  int numIterations=0;
  bool iterations_done= false;
  VectorXd x= x_;
  MatrixXd P=P_;
  MatrixXd H=H_;
  VectorXd h = VectorXd(3);
  MatrixXd K;
  MatrixXd Ht;
  MatrixXd S;
  MatrixXd Sinv;
  MatrixXd PHt;
  VectorXd y;
  float innovation;
  float stop_thres= 0.01;


  while(numIterations<maxIterations && iterations_done == false){
  std::cout<<"Iteration:"<<numIterations<<std::endl;
    h(0) = sqrt( x(0)*x(0)+ x(1)*x(1));
    if (fabs(h(0)) < 0.0001) {
      std::cout << "Division by Zero..Skipping radar measurement update" << std::endl;
      return;
    }

    h(1) = atan2( x(1) , x(0) );
    h(2) = (x(0) * x(2) + x(1) * x(3)) / h(0);
  
    H = tools.CalculateJacobian(x);

    Ht = H.transpose();
    S = H * P * Ht + R_;
    Sinv = S.inverse();
    PHt = P * Ht;
    K = PHt * Sinv;

    y =  z -  h - H* (x_- x);
    std::cout<<"y="<<y<<std::endl;
    
    if (y(1) > M_PI){ 
      y(1) = y(1) - 2*M_PI;
    }
    else if (y(1) < -M_PI) {
      y(1) = y(1) + 2*M_PI;
    }

    //new estimate
    x = x_ + (K * y);
    numIterations+=1;
    innovation=sqrt(y(0)*y(0)+ y(1)*y(1)+ y(2)*y(2));
    std::cout<<"Innovation"<<innovation<<std::endl;
    if (innovation< stop_thres) iterations_done = true;
 }
  // update covariance matrix
  x_= x;
  P_ = (I_ - K * H) * P;
}

