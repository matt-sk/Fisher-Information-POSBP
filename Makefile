CC=g++-6

default: Fisher.so Fisher

Fisher.so: FisherLibrary.cc Fisher.hh
	$(CC) -O2 -std=c++14 -shared -fPIC -o Fisher.so FisherLibrary.cc

Fisher: calcFisher.cc Fisher.hh
	$(CC) -O2 -std=c++14 -o Fisher calcFisher.cc