CXX=g++-7
CXX=g++
CXXFLAGS=-O3 -std=c++14

#all: Fisher.so Fisher.dylib Fisher
all: Fisher.so Fisher

%.so: %.so.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -fPIC $(CPPFLAGS) -o $@ $(LDLIBS) $<

%.dylib: %.so.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -dynamiclib $(CPPFLAGS) -o $@ $(LDLIBS) $<

Fisher.so: Fisher.hh

Fisher.dylib: Fisher.hh

Fisher: LDLIBS:=-lboost_program_options $(LDLIBS)
Fisher: Fisher.cc Fisher.hh
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(LDLIBS) $<
