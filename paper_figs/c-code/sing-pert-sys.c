#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "sing-pert-sys.h"
#include "constants.h"

int f(double t, const double y[], double f[], void *params) {

  double* p = (double *) params;
  double epsinv = p[0];
  double lambda = p[1];

  f[0] = y[1] - lambda*y[0];
  f[1] = y[0] - y[1]*epsinv*(1 + 10/(1.5 - sin(y[1])));

  return GSL_SUCCESS;
}

double ball_perim_eval(const double y0, const double epsinv, const double ball_radius) {
  // hold lambda and x0 constant, vary x1 and eps

  double* pars = (double *) sys->params;
  pars[0] = epsinv;

  double t = 0; // t0
  double y[2] = {CONSTANTS->X0_TRUE, y0}; // y(t0), X0 constant

  double eval = 0;
  bool success = true;
  gsl_odeiv2_driver_reset(driver);
  for(int i = 0; i < CONSTANTS->NTIMES; i++) {
    int gsl_status = gsl_odeiv2_driver_apply(driver, &t, CONSTANTS->TIMES[i], y);
    if(gsl_status == GSL_FAILURE) {
      success = false;
      printf("******************************\nfailure\n******************************\n");
    }
    eval += pow(y[1] - CONSTANTS->YDATA[i], 2);
  }

  if(success) {
    return eval - ball_radius;
  }
  else {
    return GSL_FAILURE;
  }

}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

double first_test(const double y0, const double epsinv, const double ball_radius) {
  // hold lambda and x0 constant, vary x1 and eps

  double* pars = (double *) sys->params;
  pars[0] = epsinv;

  double t = 0; // t0
  double y[2] = {CONSTANTS->X0_TRUE, y0}; // y(t0), X0 constant

  double eval = 0;
  gsl_odeiv2_driver_reset(driver);
  for(int i = 0; i < CONSTANTS->NTIMES; i++) {
    int gsl_status = gsl_odeiv2_driver_apply(driver, &t, CONSTANTS->TIMES[i], y);
    if(gsl_status == GSL_FAILURE) {
      printf("******************************\nfailure\n******************************\n");
    }
    eval += pow(y[1] - CONSTANTS->YDATA[i], 2);
    printf("y(%d) = %2.16f\n", i, y[1]);
  }

  printf("f = %2.16f\n", eval - ball_radius);

  return eval - ball_radius;

}
