#ifndef SING_PERT_SYS_H
#define SING_PERT_SYS_H

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

/* integrator-related variables */
gsl_odeiv2_system* sys;
gsl_odeiv2_driver* driver;

/* defines the ball perimeter in model space */
double ball_perim_eval(const double y0, const double epsinv, const double ball_radius);

/* defines rhs fn of ODE */
int f(double t, const double y[], double f[], void *params);

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
double first_test(const double y0, const double epsinv, const double ball_radius);
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

#endif
