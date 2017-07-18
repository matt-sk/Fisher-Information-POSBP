#include "Fisher.hh"

extern "C" float FI2_f( float t1, float t2, float p, float lambda ) {
	Fisher::Calculator<2,float> FI;

	return FI( {t1, t2}, p, lambda );
}

extern "C" double FI2( double t1, double t2, double p, double lambda ) {
	Fisher::Calculator<2,double> FI;

	return FI( {t1, t2}, p, lambda );
}

extern "C" float FI3_f( float t1, float t2, float t3, float p, float lambda ) {
	Fisher::Calculator<3,float> FI;

	return FI( {t1, t2, t3}, p, lambda );
}

extern "C" double FI3( double t1, double t2, double t3, double p, double lambda ) {
	Fisher::Calculator<3,double> FI;

	return FI( {t1, t2, t3}, p, lambda );
}

extern "C" float FI4_f( float t1, float t2, float t3, float t4, float p, float lambda ) {
	Fisher::Calculator<4,float> FI;

	return FI( {t1, t2, t3, t4}, p, lambda );
}

extern "C" double FI4( double t1, double t2, double t3, double t4, double p, double lambda ) {
	Fisher::Calculator<4,double> FI;

	return FI( {t1, t2, t3, t4}, p, lambda );
}

extern "C" float FI2_threaded_f( float t1, float t2, float p, float lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<2,float> FI( numThreads );

	return FI( {t1, t2}, p, lambda );
}

extern "C" double FI2_threaded( double t1, double t2, double p, double lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<2,double> FI( numThreads );

	return FI( {t1, t2}, p, lambda );
}

extern "C" float FI3_threaded_f( float t1, float t2, float t3, float p, float lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<3,float> FI( numThreads );

	return FI( {t1, t2, t3}, p, lambda );
}

extern "C" double FI3_threaded( double t1, double t2, double t3, double p, double lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<3,double> FI( numThreads );

	return FI( {t1, t2, t3}, p, lambda );
}

extern "C" float FI4_threaded_f( float t1, float t2, float t3, float t4, float p, float lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<4,float> FI( numThreads );

	return FI( {t1, t2, t3, t4}, p, lambda );
}

extern "C" double FI4_threaded( double t1, double t2, double t3, double t4, double p, double lambda, size_t numThreads ) {
	Fisher::ThreadedCalculator<4,double> FI( numThreads );

	return FI( {t1, t2, t3, t4}, p, lambda );
}