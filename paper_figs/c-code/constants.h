#ifndef CONSTANTS_H
#define CONSTANTS_H

/* defines various global constants :( for AUTO and related fns */
typedef struct {
  double YDATA[3];
  double TIMES[3];
  int NTIMES;

  double X0_TRUE;
  double Y0_TRUE;
  double EPSINV_TRUE;
  double LAMBDA_TRUE;
} constants;

constants* CONSTANTS;
  

#endif
