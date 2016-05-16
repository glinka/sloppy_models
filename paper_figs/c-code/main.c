#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include <math.h>

#include "constants.h"
#include "sing-pert-sys.h"

int f(double t, const double y[], double f[], void *params) {

  double* p = (double *) params;
  double epsinv = p[0];
  double lambda = p[1];

  f[0] = y[1] - lambda*y[0];
  f[1] = y[0] - y[1]*epsinv*(1 + 10/(1.5 - sin(y[1])));

  return GSL_SUCCESS;
}

int main() {
  /* do some initialization of integrator and constants */
  int ndim = 2;
  const double abstol = 1e-6;
  const double reltol = 1e-6;
  const double y_scaling = 1;
  const double dy_scaling = 1;
  control = gsl_odeiv2_control_standard_new(abstol, reltol, y_scaling, dy_scaling);

  stepper = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, ndim);

  sys = malloc(sizeof(gsl_odeiv2_system));
  sys->function = f;
  /* sys->jacobian = jac; */ // should not be needed, hopefully can use an integrator that only uses 'f'
  sys->dimension = ndim;
  double* pars = (double *) malloc(2*sizeof(double));
  pars[0] = EPSINV_TRUE;
  pars[1] = LAMBDA_TRUE;
  sys->params = (void *) pars;

  evolve = gsl_odeiv2_evolve_alloc(ndim);

  /* set up YDATA */
  YDATA = malloc(NTIMES*sizeof(double));
  
  double t = 0; // t0
  double y[2] = {X0_TRUE, Y0_TRUE}; // y(t0), X0 constant
  double stepsize = 1e-3;

  for(int i = 0; i < NTIMES; i++) {
    while(t < TIMES[i]) {
      int status = gsl_odeiv2_evolve_apply(evolve, control, stepper, sys, &t, TIMES[i], &stepsize, y);
      // add i^th term to function evaluation
    }
    YDATA[i] = y[1];
  }

  printf("%f\n%f\n%f\n", YDATA[0], YDATA[1], YDATA[2]);


  return 0;
}

// gcc -c main.c -o main.o; gcc main.o -o test -lgsl -lblas

/* int f(double t, const double y[], double f[], void *params) { */

/*   double* p = (double *) params; */
/*   double epsinv = p[0]; */
/*   double lambda = p[1]; */

/*   f[0] = y - lambda*x; */
/*   f[1] = x - y*epsinv*(1 + 10/(1.5 - sin(y))); */

/*   return GSL_SUCCESS; */
/* } */

/* int jac(double t, const double y[], double *dfdy, double dfdt[], void *params) { */
/*   double k = *(double *) params; */
/*   dfdt[0] = -k; */
/*   dfdy[0] = 0; */
/*   return GSL_SUCCESS; */
/* } */

/* int main(int argc, char** argv) { */
/*   int dim = 1; */
/*   double k = 1.0; */
/*   gsl_odeiv2_system sys = {f, jac, dim, &k}; */
/*   gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(dim); */
/*   const double abstol = 1e-6; */
/*   const double reltol = 1e-6; */
/*   const double y_scaling = 1; */
/*   const double dy_scaling = 1; */
/*   gsl_odeiv2_control* control = gsl_odeiv2_control_standard_new(abstol, reltol, y_scaling, dy_scaling); */
/*   gsl_odeiv2_step* stepper = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, dim); */
/*   double t = 0; */
/*   double tf = 1; */
/*   double h = 1e-3; */
/*   double y = 10; */
/*   printf("%f\n", y - tf*tf/2); */
/*   while(t < tf) { */
/*     int status = gsl_odeiv2_evolve_apply(evolve, control, stepper, &sys, &t, tf, &h, &y); */
/*   } */
/*   printf("%f\n", y); */
/*   printf("%f\n", t); */
/*   /\* printf("%d\n", status); *\/ */
/* } */
