CXX=g++-7
CXXFLAGS=-O3 -std=c++14

all: Fisher.so Fisher

%.so: %.so.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -fPIC $(CPPFLAGS) $(LDPATH) -o $@ $(LDFILES) $<

Fisher.so: Fisher.hh

Fisher: Fisher.hh
Fisher: LDLIBS=-lboost_program_options