#include <stdio.h>
#include <auto_f2c.h>

#include "linear-sing-pert-sys.h"
#include "linear-constants.h"

/* find level sets of 'objective fn'/'ball perimeter fn' using AUTO */

int func (integer ndim, const doublereal *u, const integer *icp, const doublereal *par, integer ijac, doublereal *f, doublereal *dfdu, doublereal *dfdp) {

  // evaluates the perimeter fn at 'u', 'par'
  // par = {epsinv, ball_radius}o
  *f = ball_perim_eval(*u, par[0], par[1]);
  if(*f == GSL_FAILURE) {
    return 1;
  }
  else {
    return 0;
  }

}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int stpnt (integer ndim, doublereal t, doublereal *u, doublereal *par) {

  // for ellipses:
  // u[0] = y0; par = {epsinv, r^2}
  /* u[0] = 2.00001; */
  /* par[0] = 1.00001; */
  /* par[1] = 3.009172445201692e-11; */

  /* for other ellipses: */
  u[0] = 2.001;
  par[0] = 1.001;
  par[1] = 3.0069575426966934e-07;

  /* // for divergence upwards */
  /* u[0] = 2.; */
  /* par[0] = 2; */
  /* par[1] = 0.2711059853722726; //0.11310559624640291; */

  return 0;

}
/* ----------------------------------------------------------------------*/
/* ----------------------------------------------------------------------*/
int pvls (integer ndim, const doublereal *u,
          doublereal *par)
{
  /* do some initialization of integrator and constants */
  sys = malloc(sizeof(gsl_odeiv2_system));
  sys->function = f;
  /* sys->jacobian = jac; */ // should not be needed, hopefully can use an integrator that only uses 'f'
  const int ode_ndim = 1;
  sys->dimension = ode_ndim;

  /* set up YDATA */
  CONSTANTS = malloc(sizeof(constants));
  CONSTANTS->NTIMES = 3;
  CONSTANTS->TIMES[0] = 1;
  CONSTANTS->TIMES[1] = 2;
  CONSTANTS->TIMES[2] = 6;
  CONSTANTS->Y0_TRUE = 2;
  // for ellipses:
  CONSTANTS->EPSINV_TRUE = 1;
  /* // for beaks: */
  /* CONSTANTS->EPSINV_TRUE = 1; */

  /* CONSTANTS->YDATA = malloc(NTIMES*sizeof(double)); */

  double* pars = (double *) malloc(sizeof(double));
  pars[0] = CONSTANTS->EPSINV_TRUE;
  sys->params = (void *) pars;
  
  double t = 0; // t0
  double y = CONSTANTS->Y0_TRUE; // y(t0), X0 constant
  double stepsize = 1e-3;

  // ste gsl driver which is needed for certain integrators
  const double abstol = 1e-12;
  const double reltol = 1e-12;
  const double y_scaling = 1;
  const double dy_scaling = 1;
  driver = gsl_odeiv2_driver_alloc_standard_new(sys, gsl_odeiv2_step_rk8pd, stepsize, abstol, reltol, y_scaling, dy_scaling); // gsl_odeiv2_step_msadams

  for(int i = 0; i < CONSTANTS->NTIMES; i++) {
    gsl_odeiv2_driver_apply(driver, &t, CONSTANTS->TIMES[i], &y);
    /* while(t < CONSTANTS->TIMES[i]) { */
    /*   gsl_odeiv2_evolve_apply(evolve, control, stepper, sys, &t, CONSTANTS->TIMES[i], &stepsize, y); */
    /*   // add i^th term to function evaluation */
    /* } */
    CONSTANTS->YDATA[i] = y;
  }

  /* first_test(2.00001, 1.00001, 3.009172445201692e-11); */

  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int bcnd (integer ndim, const doublereal *par, const integer *icp,
          integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
          doublereal *fb, doublereal *dbc)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int icnd (integer ndim, const doublereal *par, const integer *icp,
          integer nint, const doublereal *u, const doublereal *uold,
          const doublereal *udot, const doublereal *upold, integer ijac,
          doublereal *fi, doublereal *dint)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int fopt (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *fs, doublereal *dfdu, doublereal *dfdp)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
