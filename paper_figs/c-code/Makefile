SRCS=lc.c sing-pert-sys.c
LINSRCS=linearlc.c linear-sing-pert-sys.c
OBJECTS=$(SRCS:.c=.o)
LINOBJECTS=$(LINSRCS:.c=.o)

CXX = gcc

CXXFLAGS = -g -Wall -std=c99 -lgsl -lblas -O3 -I/home/alexander/local/auto/07p/include

all: linearlc_auto #lc_auto 

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< 

lc_auto: $(OBJECTS)
	gfortran -L/home/alexander/local/auto/07p/lib -fopenmp -O sing-pert-sys.o lc.o -o lc.exe /home/alexander/local/auto/07p/lib/*.o -lauto_c  -lgsl -lopenblas

linearlc_auto: $(LINOBJECTS)
	gfortran -L/home/alexander/local/auto/07p/lib -fopenmp -O linear-sing-pert-sys.o linearlc.o -o linearlc.exe /home/alexander/local/auto/07p/lib/*.o -lauto_c  -lgsl -lopenblas

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) -MM $^ > ./.depend

clean:
	$(RM) *.o 

include .depend
