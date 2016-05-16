#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>



int f(double t, const double y[], double f[], void *params) {
  double k = *(double *) params;
  f[0] = -k*t;
  return GSL_SUCCESS;
}

int jac(double t, const double y[], double *dfdy, double dfdt[], void *params) {
  double k = *(double *) params;
  dfdt[0] = -k;
  dfdy[0] = 0;
  return GSL_SUCCESS;
}

/* int main() { */
/*   double x[2] = {0,1}; */
/*   void* p = (void *) x; */
/*   double* y = (double *) p; */
/*   y[1] = 2; */
/*   printf("%f\n%f\n", x[0], x[1]); */
/* } */


int main(int argc, char** argv) {
  int dim = 1;
  double k = 1.0;
  gsl_odeiv2_system* sys = malloc(sizeof(gsl_odeiv2_system));
  sys->function = f;
  sys->jacobian = jac;
  sys->dimension = dim;
  sys->params = (void *) &k;
  /* {f, jac, dim, &k}; */
  gsl_odeiv2_evolve* evolve = gsl_odeiv2_evolve_alloc(dim);
  const double abstol = 1e-6;
  const double reltol = 1e-6;
  const double y_scaling = 1;
  const double dy_scaling = 1;
  gsl_odeiv2_control* control = gsl_odeiv2_control_standard_new(abstol, reltol, y_scaling, dy_scaling);
  gsl_odeiv2_step* stepper = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, dim);
  double t = 0;
  double tf = 1;
  double h = 1e-3;
  double y = 10;
  printf("%f\n", y - tf*tf/2);
  while(t < tf) {
    int status = gsl_odeiv2_evolve_apply(evolve, control, stepper, sys, &t, tf, &h, &y);
  }
  printf("%f\n", y);
  printf("%f\n", t);
  /* printf("%d\n", status); */
}
