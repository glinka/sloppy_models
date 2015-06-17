SRCS=main.cc unidentifiable_exp.cc
OBJECTS=$(SRCS:.cc=.o)

CXX = g++

CXXFLAGS = -g -Wall -std=c++0x -I/home/oakridge/holiday/workspace/dmaps -I/home/oakridge/holiday/workspace/newton_gmres -I/home/oakridge/holiday/workspace/util_fns -lutil_fns -leigensolvers -O3

all: dmap_sloppy_params

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $<

dmap_sloppy_params: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) -MM $^ > ./.depend

clean:
	$(RM) *.o
	$(RM) .depend

include .depend
