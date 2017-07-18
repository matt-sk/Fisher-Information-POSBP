#include <sstream>
#include <iostream>
#include <array>
#include <limits>

#include <boost/program_options.hpp>

#include "Fisher.hh"

namespace opts = boost::program_options;

int main( int argc, char* argv[] ) {

	using real_approx_t = double;

	// Declare the supported options.
	opts::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "show this help message")
		("threaded,t", opts::value<size_t>()->implicit_value(2)->default_value(0), "Number of worker threads.")
		("input-parameter", opts::value< std::vector<real_approx_t> >(), "t1, t2, [t3, t4,] p, lambda")
	;

	opts::positional_options_description posOpts;
	posOpts.add("input-parameter", -1);

	opts::variables_map optVars;
	opts::store( opts::command_line_parser(argc, argv).options(desc).style(opts::command_line_style::unix_style|opts::command_line_style::allow_long_disguise).positional(posOpts).run(), optVars );
	opts::notify( optVars ); 

	if( optVars.count("help") ) {
		std::cout << desc << std::endl;
		return 1;
	}

	auto fisherParameters = optVars["input-parameter"].as<std::vector<real_approx_t>>( );
	auto numParameters = fisherParameters.size( );

	if( numParameters > 6 ) {
		std::cerr << "Too many parameters." << std::endl;
		return 1;
	}

	if( numParameters < 3 ) {
		std::cerr << "Too few parameters." << std::endl;
		return 1;
	}

	// Output the calling parameters.
	std::clog << "t = { ";
	for( auto it = fisherParameters.cbegin(); it != fisherParameters.cend() - 3; ++it ) std::clog << *it << ", ";
	std::clog << fisherParameters[ numParameters - 3 ] << " }, p = " << fisherParameters[ numParameters - 2 ] << ", lambda = " << fisherParameters[ numParameters - 1 ] << std::endl;

	// Determine the correct class to use based on parameters, and calculate the Fisher Information requested.
	real_approx_t FI;
	auto numWorkerThreads = optVars["threaded"].as<size_t>();
	if( numWorkerThreads > 1) {
		std::clog << "Threaded computation using " << numWorkerThreads << " threads" << std::endl;
		switch( numParameters ) {
			case 6 :	FI = Fisher::ThreadedCalculator<4,real_approx_t>( numWorkerThreads ) ( { fisherParameters[0], fisherParameters[1], fisherParameters[2], fisherParameters[3] }, fisherParameters[4], fisherParameters[5] );
						break;
			case 5 :	FI = Fisher::ThreadedCalculator<3,real_approx_t>( numWorkerThreads ) ( { fisherParameters[0], fisherParameters[1], fisherParameters[2] }, fisherParameters[3], fisherParameters[4] );
						break;
			case 4 :	FI = Fisher::ThreadedCalculator<2,real_approx_t>( numWorkerThreads ) ( { fisherParameters[0], fisherParameters[1] }, fisherParameters[2], fisherParameters[3] );
						break;
			// case 3 :	FI = Fisher::ThreadedCalculator<1,real_approx_t>( numWorkerThreads ) ( { fisherParameters[0] }, fisherParameters[1], fisherParameters[2] );
		}
	} else {
		std::clog << "Sequential computation" << std::endl;
		switch( numParameters ) {
			case 6 :	FI = Fisher::Calculator<4,real_approx_t>() ( { fisherParameters[0], fisherParameters[1], fisherParameters[2], fisherParameters[3] }, fisherParameters[4], fisherParameters[5] );
						break;
			case 5 :	FI = Fisher::Calculator<3,real_approx_t>()( { fisherParameters[0], fisherParameters[1], fisherParameters[2] }, fisherParameters[3], fisherParameters[4] );
						break;
			case 4 :	FI = Fisher::Calculator<2,real_approx_t>()( { fisherParameters[0], fisherParameters[1] }, fisherParameters[2], fisherParameters[3] );
						break;
			// case 3 :	FI = Fisher::Calculator<1,real_approx_t>()( { fisherParameters[0] }, fisherParameters[1], fisherParameters[2] );
		}
	}

	// Output the computed value.
	std::cout.precision(std::numeric_limits<real_approx_t>::max_digits10);
	std::cout << FI << std::endl;

	return 0;
}