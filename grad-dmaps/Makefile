SRCS=main.cc
OBJECTS=$(SRCS:.cc=.o)

CXX = g++

CXXFLAGS = -g -Wall -std=c++0x -I/home/alexander/workspace/dmaps -I/home/alexander/workspace/newton_gmres -I/home/alexander/workspace/util_fns -I/home/alexander/local/eigen -L/home/alexander/local/lib -lutil_fns -leigensolvers -O3

all: gradient_dmaps

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $<

gradient_dmaps: $(OBJECTS)
	$(CXX) -o "gradient-dmaps" $^ $(CXXFLAGS)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CXXFLAGS) -MM $^ > ./.depend

clean:
	$(RM) *.o
	$(RM) .depend

include .depend
