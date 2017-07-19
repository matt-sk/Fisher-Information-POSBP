#include <array>
#include <unordered_map>
#include <forward_list>
#include <string>
#include <bitset>
#include <boost/dynamic_bitset.hpp>

#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>

#include <numeric>

#include <cmath>
#include <boost/math/special_functions/binomial.hpp>

#ifndef FISHER_HH_DECLARATIONS_AND_DEFINITIONS
#define FISHER_HH_DECLARATIONS_AND_DEFINITIONS

//Extend the std:: namespace to use a hash for std::array<T,N>.
namespace std
{
	template<typename T, size_t N> struct hash<std::array<T,N>>
	{
		typedef std::array<T,N> argument_type;
		typedef std::size_t result_type;
		result_type operator()(argument_type const& A) const
		{
			// std::hash<std::string> h;
			// return h( std::string(A.cbegin(), A.cend()) );

			result_type hashval = 0;
			for( auto i = 0, posval = 1; i < N-1; ++i, posval *= 10 ) {
				hashval += A[i]*posval;
			}

			return hashval;
		}
	};
}

namespace Fisher {

	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	//    Class Declarations
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	template< std::size_t> 
	class PreCalculator;

	template< std::size_t, typename > 
	class Calculator;

	struct IdxPos {
		template< std::size_t D > 
		static size_t calculate( const std::array<unsigned int,D>&, size_t );
	};

	template< std::size_t D, typename real_approx_t=double > 
	class SliceContainer { 
		public: 
			using idx_t = std::array< unsigned int, D >;
			using slice_t = std::vector< real_approx_t >;
			using size_type = typename std::unordered_map<size_t, slice_t>::size_type;

			SliceContainer( ) { }

			real_approx_t& operator[] ( const idx_t& idx ) { auto s = sliceId(idx); return data.at( s )[ IdxPos::calculate(idx, s) ]; }
			real_approx_t& at( const idx_t& idx ) { auto s = sliceId(idx); return data.at( s ).at( IdxPos::calculate(idx, s) ); }

			void insertSlice( size_t s );
			size_type eraseSlice( size_t s ) { return data.erase( s ); }
			void clear( ) noexcept { data.clear( ); }

			slice_t & getSlice( size_t s ) { return data.at(s); }

		private:
			size_t sliceId( const idx_t& idx ) { return std::accumulate( idx.cbegin(), idx.cend(), static_cast<size_t>(0) ); }
			std::unordered_map<size_t, slice_t> data;
	};

	template< std::size_t D > 
	class PreCalculator {
		public:
			template<typename real_approx_t> 
			static void preCalculate( Calculator<D,real_approx_t>*, const std::array<real_approx_t,D>&, const real_approx_t, const real_approx_t );

		private:
			template<typename real_approx_t> 
			static void doWork( Calculator<D,real_approx_t>*, const std::array< real_approx_t, D >, const std::array< real_approx_t, D >, const real_approx_t );
	};

	template< std::size_t D, typename real_approx_t=double > 
	class Calculator {
		public:
			Calculator( ) { }

			real_approx_t operator() ( const std::array<real_approx_t,D>& t, const real_approx_t p, const real_approx_t lambda );

		protected:
			using idx_t = typename SliceContainer<D,real_approx_t>::idx_t;
			using slice_t = typename SliceContainer<D,real_approx_t>::slice_t;

			SliceContainer<D,real_approx_t> L, dL_dlambda;
			std::unordered_map<idx_t,real_approx_t> Q, dQ_dlambda, P, dP_dlambda; // Maybe make these map<idx_t,real_approx_t> instead of unordered_map, since they are being iterated over, but that means writing a compaitor class.

			std::forward_list<idx_t> waitingIndices;
			void generateSliceIndices( const size_t, size_t = 0, unsigned int = 0 );

			virtual real_approx_t calculateSlice( size_t );
			virtual real_approx_t calculateIndex( const idx_t& );

		private:
			friend class PreCalculator<D>;

			size_t slice; // Currently computing slice. (May not be needed)
			real_approx_t total;

	};
	
	template< std::size_t D, typename real_approx_t=double > 
	class ThreadedCalculator : public Calculator<D,real_approx_t> {
		public:
			ThreadedCalculator( size_t numThreads = 0 );
			virtual ~ThreadedCalculator( );

		private:

			using idx_t = typename Calculator<D,real_approx_t>::idx_t;
			using slice_t = typename Calculator<D,real_approx_t>::slice_t;

			virtual real_approx_t calculateSlice( size_t );
			// virtual real_approx_t calculateIndex( const idx_t& );

			void indexCalculatorLoop( size_t, size_t );

			real_approx_t sliceDelta;
			bool workerThreadTerminateFlag;
			
			std::mutex mtx;
			std::condition_variable workerThreadWakeCond, workerThreadSliceCompletedCond;
			boost::dynamic_bitset<> workerThreadCompleteFlags;
			std::vector<std::thread> workerThreads;
	};

	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	//    Function Definitions 
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-

	// FUnctions to calculate the index of a given index within a slice.
	template< > 
	size_t IdxPos::calculate<1>( const std::array<unsigned int,1>& idx, size_t s) { 
		return 0;
	}

	template< > 
	size_t IdxPos::calculate<2>( const std::array<unsigned int,2>& idx, size_t s) { 
		return idx[0];
	}

	template< > 
	size_t IdxPos::calculate<3>( const std::array<unsigned int,3>& idx, size_t s) { 
		return (idx[0]*(2*s-idx[0]+3))/2 + idx[1];
	}

	template< > 
	size_t IdxPos::calculate<4>( const std::array<unsigned int,4>& idx, size_t s) { 
		return (idx[0]*(3*s*(s-idx[0]+4)+idx[0]*idx[0]+11-6*idx[0]))/6 + (idx[1]*(2*(s-idx[0])-idx[1]+3))/2 + idx[2];
	}


	template< std::size_t D, typename real_approx_t >
	void SliceContainer<D,real_approx_t>::insertSlice( size_t s ) { 
		constexpr auto C = boost::math::binomial_coefficient<double>;

		data.emplace( s, C(s+D-1, s) );
	}

	// PreCalculator::preCalculate( )
	template< std::size_t D > 
	template<typename real_approx_t> void PreCalculator<D>::preCalculate( Calculator<D,real_approx_t>* calc, const std::array<real_approx_t,D>& t, const real_approx_t p, const real_approx_t lambda ) {

		// Initialise the data elements.
		calc->L.clear( );
		calc->dL_dlambda.clear( );
		calc->Q.clear( );
		calc->Q.reserve( D+1 );
		calc->dQ_dlambda.clear( );
		calc->dQ_dlambda.reserve( D+1 );
		calc->P.clear( );
		calc->P.reserve( D+1 );
		calc->dP_dlambda.clear( );
		calc->dP_dlambda.reserve( D+1 );

		// Initial calcualtions to turn t[], p, and lambda from our intput into T[] and v[] needed for the pre-calculated polynoials.
		// T[i] := t[i]-t[i-1] with the understandign that t[-1]=0, and v[i] := e^(-lambda*T[i])
		std::array< real_approx_t, D > T, v;
		T[0] = t[0];
		for( auto i = 1; i < D; ++i ) T[i] = t[i] - t[i-1];
		for( auto i = 0; i < D; ++i ) v[i] = exp( -lambda*T[i] );

		// Perform the actual pre-calculation. 
		PreCalculator<D>::doWork( calc, T, v, p );
	}


	// PreCalculator::doWork( ) < Specialised for D=2 >
	template<> 
	template<typename real_approx_t> void PreCalculator<2>::doWork( Calculator<2,real_approx_t>* calc, const std::array< real_approx_t, 2 > T, const std::array< real_approx_t, 2 > v, const real_approx_t p ) {

		real_approx_t pSq = p * p;
		real_approx_t pCb = pSq * p;
		real_approx_t denominator;

		denominator = (pSq*v[0]*v[1]-pSq*v[1]-2*p*v[0]*v[1]+p*v[1]+v[0]*v[1]+p);

		calc->Q[{0,1}] = -p*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{1,0}] = -p*v[1]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1}] = v[1]*pSq*(-1+v[0])/denominator;

		calc->P[{0,0}] = v[1]*v[0]*(p-1)*(p-1)/denominator;
		calc->P[{0,1}] = -v[1]*v[0]*p*(p-1)/denominator;
		calc->P[{1,0}] = -v[1]*v[0]*p*(p-1)/denominator;
		calc->P[{1,1}] = pSq*v[0]*v[1]/denominator;

		denominator *= denominator;

		calc->dQ_dlambda[{0,1}] = p*v[1]*(p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]-T[0]*v[0]-T[1]*v[0])/denominator;
		calc->dQ_dlambda[{1,0}] = -p*v[1]*(p-1)*(p*T[0]*v[0]*v[1]-p*T[0]*v[0]-p*T[1]*v[0]-T[0]*v[0]*v[1]+p*T[1])/denominator;
		calc->dQ_dlambda[{1,1}] = v[1]*pSq*(p*T[0]*v[0]*v[1]-p*T[0]*v[0]-p*T[1]*v[0]-T[0]*v[0]*v[1]+p*T[1])/denominator;

		calc->dP_dlambda[{0,0}] = v[1]*v[0]*(p-1)*(p-1)*p*(p*T[0]*v[1]-T[0]*v[1]-T[0]-T[1])/denominator;
		calc->dP_dlambda[{0,1}] = -v[1]*v[0]*pSq*(p-1)*(p*T[0]*v[1]-T[0]*v[1]-T[0]-T[1])/denominator;
		calc->dP_dlambda[{1,0}] = -v[1]*v[0]*pSq*(p-1)*(p*T[0]*v[1]-T[0]*v[1]-T[0]-T[1])/denominator;
		calc->dP_dlambda[{1,1}] = pCb*v[0]*v[1]*(p*T[0]*v[1]-T[0]*v[1]-T[0]-T[1])/denominator;
	}

	// PreCalculator::doWork( ) < Specialised for D=3 >
	template<> 
	template<typename real_approx_t> void PreCalculator<3>::doWork( Calculator<3,real_approx_t>* calc, const std::array< real_approx_t, 3 > T, const std::array< real_approx_t, 3 > v, const real_approx_t p ) {

		real_approx_t pp = p * p;
		real_approx_t ppp = pp * p;
		real_approx_t pppp = ppp * p;

		real_approx_t denominator;

		denominator = 3*p*v[0]*v[1]*v[2]-3*pp*v[0]*v[1]*v[2]+ppp*v[0]*v[1]*v[2]-p*v[1]*v[2]+2*pp*v[1]*v[2]-ppp*v[1]*v[2]-v[0]*v[1]*v[2]-p*v[2]+pp*v[2]-p;

		calc->Q[{0,0,1}] = -p*(-2*p*v[0]*v[1]*v[2]+pp*v[0]*v[1]*v[2]+p*v[1]*v[2]-pp*v[1]*v[2]+v[0]*v[1]*v[2]+p*v[2]-1)/denominator;
		calc->Q[{0,1,0}] = -p*v[2]*(p-1)*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{0,1,1}] = v[2]*pp*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{1,0,0}] = -p*v[1]*v[2]*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,0,1}] = pp*v[1]*v[2]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,0}] = pp*v[1]*v[2]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,1}] = -v[2]*v[1]*ppp*(-1+v[0])/denominator;

		calc->P[{0,0,0}] = v[0]*v[1]*v[2]*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,0,1}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{0,1,0}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{0,1,1}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,0,0}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{1,0,1}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,1,0}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,1,1}] = -ppp*v[0]*v[1]*v[2]/denominator;

		denominator *=  denominator;

		calc->dQ_dlambda[{0,0,1}] = -p*v[2]*(-2*p*T[0]*v[0]*v[1]-2*p*T[1]*v[0]*v[1]-2*p*T[2]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]+pp*T[2]*v[0]*v[1]+p*T[1]*v[1]+p*T[2]*v[1]-pp*T[1]*v[1]-pp*T[2]*v[1]+T[0]*v[0]*v[1]+T[1]*v[0]*v[1]+T[2]*v[0]*v[1]+p*T[2])/denominator;
		calc->dQ_dlambda[{0,1,0}] = p*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]-2*p*T[1]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+pp*T[1]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[1]+p*T[1]*v[0]*v[1]+p*T[1]*v[1]*v[2]+p*T[2]*v[0]*v[1]-pp*T[0]*v[0]*v[1]-pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[0]*v[1]+T[0]*v[0]*v[1]*v[2]+T[1]*v[0]*v[1]*v[2]+pp*T[1]*v[1]+pp*T[2]*v[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{0,1,1}] = -v[2]*pp*(-2*p*T[0]*v[0]*v[1]*v[2]-2*p*T[1]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+pp*T[1]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[1]+p*T[1]*v[0]*v[1]+p*T[1]*v[1]*v[2]+p*T[2]*v[0]*v[1]-pp*T[0]*v[0]*v[1]-pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[0]*v[1]+T[0]*v[0]*v[1]*v[2]+T[1]*v[0]*v[1]*v[2]+pp*T[1]*v[1]+pp*T[2]*v[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,0,0}] = -p*v[1]*v[2]*(p-1)*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,0,1}] = pp*v[1]*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,1,0}] = pp*v[1]*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,1,1}] = -v[2]*v[1]*ppp*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;

		calc->dP_dlambda[{0,0,0}] = v[0]*v[1]*v[2]*(p-1)*(p-1)*(p-1)*p*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{0,0,1}] = -v[0]*v[1]*v[2]*pp*(p-1)*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{0,1,0}] = -v[0]*v[1]*v[2]*pp*(p-1)*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{0,1,1}] = v[0]*v[1]*v[2]*ppp*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{1,0,0}] = -v[0]*v[1]*v[2]*pp*(p-1)*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{1,0,1}] = v[0]*v[1]*v[2]*ppp*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{1,1,0}] = v[0]*v[1]*v[2]*ppp*(p-1)*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
		calc->dP_dlambda[{1,1,1}] = -pppp*v[0]*v[1]*v[2]*(-2*p*T[0]*v[1]*v[2]+pp*T[0]*v[1]*v[2]-p*T[0]*v[2]-p*T[1]*v[2]+T[0]*v[1]*v[2]+T[0]*v[2]+T[1]*v[2]+T[0]+T[1]+T[2])/denominator;
	}

	// PreCalculator::doWork( ) < Specialised for D=4 >
	template<> 
	template<typename real_approx_t> void PreCalculator<4>::doWork( Calculator<4,real_approx_t>* calc, const std::array< real_approx_t, 4 > T, const std::array< real_approx_t, 4 > v, const real_approx_t p ) {

		real_approx_t pp = p * p;
		real_approx_t ppp = pp * p;
		real_approx_t pppp = ppp * p;
		real_approx_t ppppp = pppp * p;

		real_approx_t denominator;

		denominator = -4*p*v[0]*v[1]*v[2]*v[3]+6*pp*v[0]*v[1]*v[2]*v[3]-4*ppp*v[0]*v[1]*v[2]*v[3]+pppp*v[0]*v[1]*v[2]*v[3]+p*v[1]*v[2]*v[3]-3*pp*v[1]*v[2]*v[3]+3*ppp*v[1]*v[2]*v[3]-pppp*v[1]*v[2]*v[3]+v[0]*v[1]*v[2]*v[3]+p*v[2]*v[3]-2*pp*v[2]*v[3]+ppp*v[2]*v[3]+p*v[3]-pp*v[3]+p;

		calc->Q[{0,0,0,1}] = -p*(3*p*v[0]*v[1]*v[2]*v[3]-3*pp*v[0]*v[1]*v[2]*v[3]+ppp*v[0]*v[1]*v[2]*v[3]-p*v[1]*v[2]*v[3]+2*pp*v[1]*v[2]*v[3]-ppp*v[1]*v[2]*v[3]-v[0]*v[1]*v[2]*v[3]-p*v[2]*v[3]+pp*v[2]*v[3]-p*v[3]+1)/denominator;
		calc->Q[{0,0,1,0}] = -p*v[3]*(p-1)*(-2*p*v[0]*v[1]*v[2]+pp*v[0]*v[1]*v[2]+p*v[1]*v[2]-pp*v[1]*v[2]+v[0]*v[1]*v[2]+p*v[2]-1)/denominator;
		calc->Q[{0,0,1,1}] = v[3]*pp*(-2*p*v[0]*v[1]*v[2]+pp*v[0]*v[1]*v[2]+p*v[1]*v[2]-pp*v[1]*v[2]+v[0]*v[1]*v[2]+p*v[2]-1)/denominator;
		calc->Q[{0,1,0,0}] = -p*v[2]*v[3]*(p-1)*(p-1)*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{0,1,0,1}] = pp*v[2]*v[3]*(p-1)*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{0,1,1,0}] = pp*v[2]*v[3]*(p-1)*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{0,1,1,1}] = -v[2]*v[3]*ppp*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{1,0,0,0}] = -p*v[1]*v[2]*v[3]*(p-1)*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,0,0,1}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,0,1,0}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,0,1,1}] = -ppp*v[1]*v[2]*v[3]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,0,0}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,1,0,1}] = -ppp*v[1]*v[2]*v[3]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,1,0}] = -ppp*v[1]*v[2]*v[3]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,1,1}] = v[1]*v[2]*v[3]*pppp*(-1+v[0])/denominator;

		calc->P[{0,0,0,0}] = v[0]*v[1]*v[2]*v[3]*(p-1)*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,0,0,1}] = -v[0]*v[1]*v[2]*v[3]*p*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,0,1,0}] = -v[0]*v[1]*v[2]*v[3]*p*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,0,1,1}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{0,1,0,0}] = -v[0]*v[1]*v[2]*v[3]*p*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,1,0,1}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{0,1,1,0}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{0,1,1,1}] = -v[0]*v[1]*v[2]*v[3]*ppp*(p-1)/denominator;
		calc->P[{1,0,0,0}] = -v[0]*v[1]*v[2]*v[3]*p*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{1,0,0,1}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{1,0,1,0}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{1,0,1,1}] = -v[0]*v[1]*v[2]*v[3]*ppp*(p-1)/denominator;
		calc->P[{1,1,0,0}] = v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)/denominator;
		calc->P[{1,1,0,1}] = -v[0]*v[1]*v[2]*v[3]*ppp*(p-1)/denominator;
		calc->P[{1,1,1,0}] = -v[0]*v[1]*v[2]*v[3]*ppp*(p-1)/denominator;
		calc->P[{1,1,1,1}] = pppp*v[0]*v[1]*v[2]*v[3]/denominator;

		denominator *=  denominator;

		calc->dQ_dlambda[{0,0,0,1}] = p*v[3]*(3*p*T[0]*v[0]*v[1]*v[2]+3*p*T[1]*v[0]*v[1]*v[2]+3*p*T[2]*v[0]*v[1]*v[2]+3*p*T[3]*v[0]*v[1]*v[2]-3*pp*T[0]*v[0]*v[1]*v[2]-3*pp*T[1]*v[0]*v[1]*v[2]-3*pp*T[2]*v[0]*v[1]*v[2]-3*pp*T[3]*v[0]*v[1]*v[2]+ppp*T[0]*v[0]*v[1]*v[2]+ppp*T[1]*v[0]*v[1]*v[2]+ppp*T[2]*v[0]*v[1]*v[2]+ppp*T[3]*v[0]*v[1]*v[2]-p*T[1]*v[1]*v[2]-p*T[2]*v[1]*v[2]-p*T[3]*v[1]*v[2]+2*pp*T[1]*v[1]*v[2]+2*pp*T[2]*v[1]*v[2]+2*pp*T[3]*v[1]*v[2]-ppp*T[1]*v[1]*v[2]-ppp*T[2]*v[1]*v[2]-ppp*T[3]*v[1]*v[2]-T[0]*v[0]*v[1]*v[2]-T[1]*v[0]*v[1]*v[2]-T[2]*v[0]*v[1]*v[2]-T[3]*v[0]*v[1]*v[2]-p*T[2]*v[2]-p*T[3]*v[2]+pp*T[2]*v[2]+pp*T[3]*v[2]-p*T[3])/denominator;
		calc->dQ_dlambda[{0,0,1,0}] = -p*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]+3*p*T[2]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[2]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[2]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[2]-p*T[1]*v[0]*v[1]*v[2]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[2]-p*T[2]*v[1]*v[2]*v[3]-p*T[3]*v[0]*v[1]*v[2]+2*pp*T[0]*v[0]*v[1]*v[2]+2*pp*T[1]*v[0]*v[1]*v[2]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[2]+2*pp*T[2]*v[1]*v[2]*v[3]+2*pp*T[3]*v[0]*v[1]*v[2]-ppp*T[0]*v[0]*v[1]*v[2]-ppp*T[1]*v[0]*v[1]*v[2]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[2]-ppp*T[2]*v[1]*v[2]*v[3]-ppp*T[3]*v[0]*v[1]*v[2]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-T[2]*v[0]*v[1]*v[2]*v[3]-p*T[2]*v[2]*v[3]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[1]*v[2]+pp*T[2]*v[2]*v[3]-pp*T[3]*v[1]*v[2]+ppp*T[1]*v[1]*v[2]+ppp*T[2]*v[1]*v[2]+ppp*T[3]*v[1]*v[2]-pp*T[2]*v[2]-pp*T[3]*v[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{0,0,1,1}] = v[3]*pp*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]+3*p*T[2]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[2]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[2]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[2]-p*T[1]*v[0]*v[1]*v[2]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[2]-p*T[2]*v[1]*v[2]*v[3]-p*T[3]*v[0]*v[1]*v[2]+2*pp*T[0]*v[0]*v[1]*v[2]+2*pp*T[1]*v[0]*v[1]*v[2]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[2]+2*pp*T[2]*v[1]*v[2]*v[3]+2*pp*T[3]*v[0]*v[1]*v[2]-ppp*T[0]*v[0]*v[1]*v[2]-ppp*T[1]*v[0]*v[1]*v[2]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[2]-ppp*T[2]*v[1]*v[2]*v[3]-ppp*T[3]*v[0]*v[1]*v[2]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-T[2]*v[0]*v[1]*v[2]*v[3]-p*T[2]*v[2]*v[3]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[1]*v[2]+pp*T[2]*v[2]*v[3]-pp*T[3]*v[1]*v[2]+ppp*T[1]*v[1]*v[2]+ppp*T[2]*v[1]*v[2]+ppp*T[3]*v[1]*v[2]-pp*T[2]*v[2]-pp*T[3]*v[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{0,1,0,0}] = p*v[2]*v[3]*(p-1)*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[3]-p*T[1]*v[0]*v[1]*v[3]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[3]+2*pp*T[0]*v[0]*v[1]*v[3]+2*pp*T[1]*v[0]*v[1]*v[3]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[3]-ppp*T[0]*v[0]*v[1]*v[3]-ppp*T[1]*v[0]*v[1]*v[3]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]-p*T[1]*v[0]*v[1]-p*T[2]*v[0]*v[1]-p*T[3]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[3]+pp*T[2]*v[0]*v[1]-pp*T[2]*v[1]*v[3]+pp*T[3]*v[0]*v[1]+ppp*T[1]*v[1]*v[3]+ppp*T[2]*v[1]*v[3]+p*T[2]*v[3]-pp*T[1]*v[1]-pp*T[2]*v[1]-pp*T[2]*v[3]-pp*T[3]*v[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{0,1,0,1}] = -pp*v[2]*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[3]-p*T[1]*v[0]*v[1]*v[3]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[3]+2*pp*T[0]*v[0]*v[1]*v[3]+2*pp*T[1]*v[0]*v[1]*v[3]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[3]-ppp*T[0]*v[0]*v[1]*v[3]-ppp*T[1]*v[0]*v[1]*v[3]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]-p*T[1]*v[0]*v[1]-p*T[2]*v[0]*v[1]-p*T[3]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[3]+pp*T[2]*v[0]*v[1]-pp*T[2]*v[1]*v[3]+pp*T[3]*v[0]*v[1]+ppp*T[1]*v[1]*v[3]+ppp*T[2]*v[1]*v[3]+p*T[2]*v[3]-pp*T[1]*v[1]-pp*T[2]*v[1]-pp*T[2]*v[3]-pp*T[3]*v[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{0,1,1,0}] = -pp*v[2]*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[3]-p*T[1]*v[0]*v[1]*v[3]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[3]+2*pp*T[0]*v[0]*v[1]*v[3]+2*pp*T[1]*v[0]*v[1]*v[3]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[3]-ppp*T[0]*v[0]*v[1]*v[3]-ppp*T[1]*v[0]*v[1]*v[3]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]-p*T[1]*v[0]*v[1]-p*T[2]*v[0]*v[1]-p*T[3]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[3]+pp*T[2]*v[0]*v[1]-pp*T[2]*v[1]*v[3]+pp*T[3]*v[0]*v[1]+ppp*T[1]*v[1]*v[3]+ppp*T[2]*v[1]*v[3]+p*T[2]*v[3]-pp*T[1]*v[1]-pp*T[2]*v[1]-pp*T[2]*v[3]-pp*T[3]*v[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{0,1,1,1}] = v[2]*v[3]*ppp*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]+3*p*T[1]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[1]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]*v[3]-p*T[1]*v[0]*v[1]*v[3]-p*T[1]*v[1]*v[2]*v[3]-p*T[2]*v[0]*v[1]*v[3]+2*pp*T[0]*v[0]*v[1]*v[3]+2*pp*T[1]*v[0]*v[1]*v[3]+2*pp*T[1]*v[1]*v[2]*v[3]+2*pp*T[2]*v[0]*v[1]*v[3]-ppp*T[0]*v[0]*v[1]*v[3]-ppp*T[1]*v[0]*v[1]*v[3]-ppp*T[1]*v[1]*v[2]*v[3]-ppp*T[2]*v[0]*v[1]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-T[1]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[1]-p*T[1]*v[0]*v[1]-p*T[2]*v[0]*v[1]-p*T[3]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[3]+pp*T[2]*v[0]*v[1]-pp*T[2]*v[1]*v[3]+pp*T[3]*v[0]*v[1]+ppp*T[1]*v[1]*v[3]+ppp*T[2]*v[1]*v[3]+p*T[2]*v[3]-pp*T[1]*v[1]-pp*T[2]*v[1]-pp*T[2]*v[3]-pp*T[3]*v[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,0,0,0}] = -p*v[1]*v[2]*v[3]*(p-1)*(p-1)*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,0,0,1}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,0,1,0}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,0,1,1}] = -ppp*v[1]*v[2]*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,1,0,0}] = pp*v[1]*v[2]*v[3]*(p-1)*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,1,0,1}] = -ppp*v[1]*v[2]*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,1,1,0}] = -ppp*v[1]*v[2]*v[3]*(p-1)*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;
		calc->dQ_dlambda[{1,1,1,1}] = v[1]*v[2]*v[3]*pppp*(3*p*T[0]*v[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[2]*v[3]-p*T[1]*v[0]*v[2]*v[3]+2*pp*T[0]*v[0]*v[2]*v[3]+2*pp*T[1]*v[0]*v[2]*v[3]-ppp*T[0]*v[0]*v[2]*v[3]-ppp*T[1]*v[0]*v[2]*v[3]-T[0]*v[0]*v[1]*v[2]*v[3]-p*T[0]*v[0]*v[3]-p*T[1]*v[0]*v[3]+p*T[1]*v[2]*v[3]-p*T[2]*v[0]*v[3]+pp*T[0]*v[0]*v[3]+pp*T[1]*v[0]*v[3]-2*pp*T[1]*v[2]*v[3]+pp*T[2]*v[0]*v[3]+ppp*T[1]*v[2]*v[3]-p*T[0]*v[0]-p*T[1]*v[0]+p*T[1]*v[3]-p*T[2]*v[0]+p*T[2]*v[3]-p*T[3]*v[0]-pp*T[1]*v[3]-pp*T[2]*v[3]+p*T[1]+p*T[2]+p*T[3])/denominator;

		calc->dP_dlambda[{0,0,0,0}] = v[0]*v[1]*v[2]*v[3]*(p-1)*(p-1)*(p-1)*(p-1)*p*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,0,0,1}] = -v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,0,1,0}] = -v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,0,1,1}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,1,0,0}] = -v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,1,0,1}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,1,1,0}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{0,1,1,1}] = -v[0]*v[1]*v[2]*v[3]*pppp*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,0,0,0}] = -v[0]*v[1]*v[2]*v[3]*pp*(p-1)*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,0,0,1}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,0,1,0}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,0,1,1}] = -v[0]*v[1]*v[2]*v[3]*pppp*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,1,0,0}] = v[0]*v[1]*v[2]*v[3]*ppp*(p-1)*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,1,0,1}] = -v[0]*v[1]*v[2]*v[3]*pppp*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,1,1,0}] = -v[0]*v[1]*v[2]*v[3]*pppp*(p-1)*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
		calc->dP_dlambda[{1,1,1,1}] = ppppp*v[0]*v[1]*v[2]*v[3]*(3*p*T[0]*v[1]*v[2]*v[3]-3*pp*T[0]*v[1]*v[2]*v[3]+ppp*T[0]*v[1]*v[2]*v[3]+2*p*T[0]*v[2]*v[3]+2*p*T[1]*v[2]*v[3]-pp*T[0]*v[2]*v[3]-pp*T[1]*v[2]*v[3]-T[0]*v[1]*v[2]*v[3]+p*T[0]*v[3]+p*T[1]*v[3]+p*T[2]*v[3]-T[0]*v[2]*v[3]-T[1]*v[2]*v[3]-T[0]*v[3]-T[1]*v[3]-T[2]*v[3]-T[0]-T[1]-T[2]-T[3])/denominator;
	}

	// Calculator::operator( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t Calculator<D,real_approx_t>::operator() ( const std::array<real_approx_t,D>& t, const real_approx_t p, const real_approx_t lambda ) {
		PreCalculator<D>::preCalculate( this, t, p, lambda );

		// For slice from 1 while precision not exhausted ...
		real_approx_t fisherInformation = static_cast<real_approx_t>(0);
		real_approx_t runningTotal = static_cast<real_approx_t>(-1);

		for( size_t s = 0; fisherInformation != runningTotal; ++s ) {
			runningTotal = fisherInformation;
			fisherInformation = runningTotal + calculateSlice( s );
		}

		return fisherInformation;
	}

	// Calculator::calculateSlice( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t Calculator<D,real_approx_t>::calculateSlice( size_t s )  {

		// Remove slice that is no longer needed.
		L.eraseSlice( s-(D+1) );
		dL_dlambda.eraseSlice( s-(D+1) );

		// Insert new slice. (This should have no effect on already existing slices).
		L.insertSlice( s );
		dL_dlambda.insertSlice( s );

		// Generate indices for new siice.
		waitingIndices.clear( );
		generateSliceIndices( s );
		
		// Calculate the Fisher Information delta for this slice by accumulating the delta for each index.
		// Note that the calculateIndex( ) function appends the values into L and dL_dlambda, so we don't need to do so here.
		real_approx_t runningTotal = static_cast<real_approx_t>( 0 );
		while( ! waitingIndices.empty() ) {
			runningTotal += calculateIndex( waitingIndices.front() );
			waitingIndices.pop_front( );
		}
		return runningTotal;
	}

	// Calculator::generateSliceIndices
	template< std::size_t D, typename real_approx_t >
	void Calculator<D,real_approx_t>::generateSliceIndices( const size_t s, size_t pos, unsigned int culm )  {
		
		if( pos == 0 ) waitingIndices.emplace_front( );

		if( pos < D - 2 ) {
			for( unsigned int i = 0; i <= s - culm; ++i ) {
				waitingIndices.front()[pos] = i;
				generateSliceIndices( s, pos+1, culm + i );
			}
		} else {
			for( unsigned int i = 0; i <= s - culm; ++i ) {
				waitingIndices.front()[pos] = i;
				waitingIndices.front()[pos+1] = s - culm - i;
				waitingIndices.emplace_front( waitingIndices.front() );
			}
		}

		if( pos == 0 ) waitingIndices.pop_front( ); // There will be an extra, unused index to be removed.
	}

	// Calculator::calculateIndex( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t Calculator<D,real_approx_t>::calculateIndex( const idx_t& idx ) {

		constexpr auto ZERO = static_cast<real_approx_t>(0);

		// Initial Values from P polynomial are initialised into the appropriate indices for L and dL_dlambda. We check to see if they are there, and if not we begin computation with 0.
		real_approx_t LI, dLI_dlambda;

		LI = dLI_dlambda = ZERO;
		{ auto it = P.find( idx ); if( it != P.cend() ) LI = it->second; }
		{ auto it = dP_dlambda.find( idx ); if( it != dP_dlambda.cend() ) dLI_dlambda = it->second; }

		// For each q[i,j], we calculate q[i,j, ...]*L[a-i, b-j, ...], skipping cases where any of the a-i, b-j, ... are less than zero.
		for( auto&& q : Q ) {
			idx_t prev_Idx;
			size_t i;

			for( i = 0; i < D; ++i ) {
				if( idx[i] < q.first[i] ) break;
				prev_Idx[i] = idx[i] - q.first[i];
			}
			if (i != D) continue;

			auto L_prev = L[prev_Idx];
			LI -= (q.second)*L_prev;
			dLI_dlambda -= dQ_dlambda[q.first]*L_prev + (q.second)*dL_dlambda[prev_Idx];
		}

		// Update the values in the L and dL_dlambda containers.
		L[idx] = LI;
		dL_dlambda[idx] = dLI_dlambda;

		// Return the result, skipping any 0 denominators. (Returning 0 for a 0 denominator effecitvely means this entire calculation does not affect the running total of the Fisher Information)
		return ( LI == ZERO ) ? ZERO : dLI_dlambda * dLI_dlambda / LI;
	}

	// ThreadedCalculator::ThreadedCalculator( )
	template< std::size_t D, typename real_approx_t >
	ThreadedCalculator<D,real_approx_t>::ThreadedCalculator( size_t numThreads ) 
		: workerThreadCompleteFlags( numThreads ), workerThreadTerminateFlag( false )
	{ 

		constexpr auto threadLoopFn = &ThreadedCalculator<D,real_approx_t>::indexCalculatorLoop;

		workerThreads.reserve( numThreads ); 
		for( auto i = 0; i < numThreads; ++i ) {
			workerThreads.emplace_back( threadLoopFn, this, i, numThreads );
		}

		// Wait for workers to finish initial setup.
		{
			std::unique_lock<std::mutex> lck( mtx );
			workerThreadSliceCompletedCond.wait( lck, [this]{ return workerThreadCompleteFlags.all(); } );
		}

	}

	template< std::size_t D, typename real_approx_t >
	ThreadedCalculator<D,real_approx_t>::~ThreadedCalculator( ) {

		// Set the termination flag so the workers know to terminate.
		workerThreadTerminateFlag = true;

		// Wake he worker threads.
		workerThreadWakeCond.notify_all();

		// Wait for the worker threads to terminate
		for( auto&& thrd : workerThreads ) {
			if( thrd.joinable() ) thrd.join();
		}
	}


	// ThreadedCalculator::calculateSlice( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t ThreadedCalculator<D,real_approx_t>::calculateSlice( size_t s )  {

		using calc = Calculator<D,real_approx_t>;

		// Remove slice that is no longer needed.
		calc::L.eraseSlice( s-(D+1) );
		calc::dL_dlambda.eraseSlice( s-(D+1) );

		// Insert new slice. (This should have no effect on already existing slices).
		calc::L.insertSlice( s );
		calc::dL_dlambda.insertSlice( s );

		// Generate indices for new siice as an asynchronous task.
		calc::waitingIndices.clear( );
		calc::generateSliceIndices( s );

		// Initialise the delta value for this slice.
		sliceDelta = static_cast<real_approx_t>(0);

		// Initialise the worker thread completion flags.
		workerThreadCompleteFlags.reset();

		// Notify workers to start working.
		workerThreadWakeCond.notify_all();

		// Wait for workers to finish.
		{
			std::unique_lock<std::mutex> lck( mtx );
			workerThreadSliceCompletedCond.wait( lck, [this]{ return workerThreadCompleteFlags.all(); } );
		}

		// sliceDelta is automatically accumulated by the indexCalculatorLoop( ) threads.
		return sliceDelta;
	}

	// template< std::size_t D, typename real_approx_t >
	// void ThreadedCalculator<D,real_approx_t>::indexCalculatorLoop( size_t ID ) {

	// 	using calc = Calculator<D,real_approx_t>;

	// 	real_approx_t localTotal;

	// 	std::unique_lock<std::mutex> lck( mtx ); // Lock is locked on construction.
	// 	do {
	// 		// (Re-) Initialise local Total.
	// 		localTotal = static_cast<real_approx_t>(0);

	// 		// Wait for a slice computation to start
	// 		workerThreadWakeCond.wait( lck );

	// 		while( ! calc::waitingIndices.empty() ) {
	// 			// Get the first index.
	// 			idx_t idx = std::move( calc::waitingIndices.front( ) );
	// 			calc::waitingIndices.pop_front( );

	// 			// We no longer need exclusive access, so release ownership of the mutex.
	// 			lck.unlock( );

	// 			// Calculate the Fisher information component from the index, adding it to our local total.
	// 			localTotal += calc::calculateIndex( idx );
	// 			// NOTE: calculateIndex will update L and dL_dlambda This /should/ be OK because each thread will be updating an entirely different memory area.
	// 			// However, this may cause some problems with updating size/length variables.

	// 			// Reaquire ownership of the mutex before checking emptiness again.
	// 			lck.lock( );
	// 		}

	// 		// Update 
	// 		sliceDelta += localTotal;
	// 		workerThreadCompleteFlags.set( ID, true );

	// 		workerThreadSliceCompletedCond.notify_all();
	// 	} while( true );

	// }

	template< std::size_t D, typename real_approx_t >
	void ThreadedCalculator<D,real_approx_t>::indexCalculatorLoop( size_t ID, size_t numThreads ) {

		using calc = Calculator<D,real_approx_t>;

		real_approx_t localTotal;

		std::unique_lock<std::mutex> lck( mtx ); // Lock is locked on construction.
		do {

			// Set the complete flag and notify completion. 
			// Note that we use this mechanism to indicate initial setup as well as slice completion
			workerThreadCompleteFlags.set( ID, true );
			workerThreadSliceCompletedCond.notify_all();

			// Wait for a slice computation to start
			workerThreadWakeCond.wait( lck );

			// Check for termination.
			if( workerThreadTerminateFlag ) return;

			// Release the mutex (we only need read access to the waitingIndices container for computation purposes)
			lck.unlock( );

			// Initialise local slice sub total.
			localTotal = static_cast<real_approx_t>(0);

// ***** CHANGE THIS TO USE A DEQUE INSTEAD OF A LIST *******

			// Calculate the slice stride delta.
			auto idxIt = calc::waitingIndices.cbegin();
			for( auto i = 0; (i < ID) && (idxIt != calc::waitingIndices.cend()); ++i) ++idxIt;

			while( idxIt != calc::waitingIndices.cend() ) {
				localTotal += calc::calculateIndex( *idxIt );

				for( auto i = 0; (i < numThreads) && (idxIt != calc::waitingIndices.cend()); ++i ) ++idxIt;
			}

			// Update the global slice delta.
			lck.lock( );
			sliceDelta += localTotal;

		} while( true );

	}


} // END namespace Fisher

#endif // (FISHER_HH_DECLARATIONS_AND_DEFINITIONS)