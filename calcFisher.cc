#include <sstream>
#include <iostream>
#include <array>
#include <limits>

#include "Fisher.hh"

template< typename T >
T FI( T t1, T t2, T p, T lambda ) {
	Fisher::Calculator<2,T> FI;

	return FI( {t1, t2}, p, lambda );
}

template< typename T >
T FI_threaded( T t1, T t2, T p, T lambda, int numThreads = 2 ) {
	Fisher::ThreadedCalculator<2,T> FI( numThreads );

	return FI( {t1, t2}, p, lambda );
}

template< typename T >
T FI( T t1, T t2, T t3, T p, T lambda ) {
	Fisher::Calculator<3,T> FI;

	return FI( {t1, t2, t3}, p, lambda );
}

template<typename T, size_t N>
std::ostream& operator<<( std::ostream& stream, const std::array<T,N>& A ) {
	stream << '[' << A[0];
	for( auto i = 1; i < N; ++i ) stream << ',' << A[i];
	stream << ']';
	return stream;
}

int main( int argc, char* argv[] ) {

	double t1, t2, p, lambda;
	{
		std::istringstream( argv[1] ) >> t1;
		std::istringstream( argv[2] ) >> t2;
		std::istringstream( argv[3] ) >> p;
		std::istringstream( argv[4] ) >> lambda;
	}

	std::cout.precision(std::numeric_limits<double>::max_digits10);
	std::cout << "[" << t1 << ", " << t2 << "], " << p << ", " << lambda << std::endl;

	std::cout << FI( t1, t2, p, lambda ) << std::endl;
}