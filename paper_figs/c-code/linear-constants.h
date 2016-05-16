#ifndef LINCONSTANTS_H
#define LINCONSTANTS_H

/* defines various global constants :( for AUTO and related fns */
typedef struct {
  double YDATA[3];
  double TIMES[3];
  int NTIMES;

  double Y0_TRUE;
  double EPSINV_TRUE;
} constants;

constants* CONSTANTS;
  

#endif
