#include <array>
#include <unordered_map>
#include <deque>
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
		result_type operator() (argument_type const& A) const
		{
			result_type hashval = 0;
			for( auto i = 0, posval = 1; i < N; ++i, posval *= 2 ) {
				hashval += A[i]*posval;
			}

			return hashval;
		}
	};
}

namespace Fisher {
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	//    Type alises.
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	using idx_coord_t = unsigned int;

	template< std::size_t D >
	using index_t = std::array< idx_coord_t, D >;


	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	//    Class Declarations
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	template< std::size_t> 
	class PreCalculator;

	template< std::size_t, typename > 
	class Calculator;

	template< std::size_t D >
	class IndexGenerator {
		public:
			using idx_t = index_t<D>;
			using size_type = typename std::deque<idx_t>::size_type;

			IndexGenerator( ) : slice(0) { }

			void generate( const size_t slice );
			idx_t& operator[]( const size_t i ) { return indices[i]; }
			idx_t& at( const size_t i ) { return indices.at(i); }

			auto size() { return indices.size(); }

		private:
			size_t slice;
			std::deque<idx_t> indices;

			void generateSliceIndices( size_t, size_t );
			typename std::deque<idx_t>::iterator currentGeneratingIndex;
	};

	template< >
	class IndexGenerator<2> {
		public:
			using idx_t = index_t<2>;

			IndexGenerator( ) : _slice(0), _size(0) { }

			void generate( const size_t s )  { _slice = s; _size = s+1; } // Number of indices for slice s when n=2 is always s+1
			idx_t operator[]( size_t i ) { return { static_cast<idx_coord_t>(i), static_cast<idx_coord_t>(_slice-i)}; }

			size_t size() { return _size; }

		private:
			size_t _slice, _size;
	};

	struct IdxPos {
		template< std::size_t D > 
		static size_t calculate( const typename IndexGenerator<D>::idx_t&, size_t );
	};

	template< std::size_t D, typename real_approx_t=double > 
	class SliceContainer { 
		public: 
			using idx_t = index_t<D>;
			using slice_t = std::vector< real_approx_t >;
			using size_type = typename std::array< slice_t, D+1 >::size_type;

			SliceContainer( ) { }

			// Indexing for specific elements.
			real_approx_t& operator[] ( const idx_t &idx ) { auto s = sliceId(idx); return data[ s % (D+1) ][ IdxPos::calculate<D>(idx, s) ]; }
			real_approx_t& at( const idx_t &idx ) { auto s = sliceId(idx); return data.at( s % (D+1) ).at( IdxPos::calculate<D>(idx, s) ); }

			// Indexing for slices.
			slice_t& operator[] ( const size_type s ) { return data[ s % (D+1) ]; }
			slice_t& at( const size_type s ) { return data.at( s % (D+1) ); }

			auto size() { return data.size(); }
			void clear() { for( auto slice : data ) slice.clear(); }

		private:
			size_t sliceId( const idx_t& idx ) { return std::accumulate( idx.cbegin(), idx.cend(), static_cast<size_t>(0) ); }
			std::array< slice_t, D+1 > data;
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

			IndexGenerator<D> waitingIndices;

			virtual void preCalculate( const std::array<real_approx_t,D> &t, const real_approx_t p, const real_approx_t lambda ) { PreCalculator<D>::preCalculate( this, t, p, lambda ); }
			virtual void initialiseSlice( );

			virtual real_approx_t calculateSlice( );
			
			real_approx_t calculateIndex( const idx_t& );

			size_t slice; // Currently computing slice.

		private:
			friend class PreCalculator<D>;

	};
	
	template< std::size_t D, typename real_approx_t=double > 
	class ThreadedCalculator : public Calculator<D,real_approx_t> {
		public:
			ThreadedCalculator( size_t numThreads = 0 );
			virtual ~ThreadedCalculator( );

		private:

			using idx_t = typename Calculator<D,real_approx_t>::idx_t;
			using slice_t = typename Calculator<D,real_approx_t>::slice_t;

			virtual void preCalculate( const std::array<real_approx_t,D>&, const real_approx_t, const real_approx_t );
			virtual real_approx_t calculateSlice( );

			void indexCalculatorLoop( size_t, size_t );

			real_approx_t sliceDelta;
			bool workerThreadTerminateFlag;
			
			std::mutex mtx;
			std::condition_variable workerThreadWakeCond, workerThreadSliceCompletedCond;
			boost::dynamic_bitset<> workerThreadCompleteFlags;
			std::vector<std::thread> workerThreads;
			std::future<void> sliceInitTask;
	};

	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-
	//    Function Definitions 
	// -= =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= =-

	// FUnctions to calculate the index of a given index within a slice.
	template< > 
	size_t IdxPos::calculate<1>( const typename IndexGenerator<1>::idx_t &idx, size_t s ) { 
		return 0;
	}

	template< > 
	size_t IdxPos::calculate<2>( const typename IndexGenerator<2>::idx_t &idx, size_t s ) { 
		return idx[0];
	}

	template< > 
	size_t IdxPos::calculate<3>( const typename IndexGenerator<3>::idx_t &idx, size_t s ) { 
		return (idx[0]*(2*s-idx[0]+3))/2 + idx[1];
	}

	template< > 
	size_t IdxPos::calculate<4>( const typename IndexGenerator<4>::idx_t &idx, size_t s ) { 
		return (idx[0]*(3*s*(s-idx[0]+4)+idx[0]*idx[0]+11-6*idx[0]))/6 + (idx[1]*(2*(s-idx[0])-idx[1]+3))/2 + idx[2];
	}

	// Index Generator methods.
	template< std::size_t D >
	void IndexGenerator<D>::generate( const size_t s ) { 

		constexpr auto C = boost::math::binomial_coefficient<double>;

		slice = s; 
		indices.resize( C( s+D-1, s ) );

		currentGeneratingIndex = indices.begin();
		generateSliceIndices( 0, 0 );
	}

	template< std::size_t D >
	void IndexGenerator<D>::generateSliceIndices( size_t pos, size_t culm )  {
		
		if( pos < D - 2 ) {
			for( auto i = 0; i <= slice - culm; ++i ) {
				(*currentGeneratingIndex)[pos] = i;
				generateSliceIndices( pos+1, culm + i );
			}
		} else {
			for( auto i = 0; i <= slice - culm; ++i ) {
				(*currentGeneratingIndex)[pos] = i;
				(*currentGeneratingIndex)[pos+1] = slice - culm - i;
				++currentGeneratingIndex;
				if( currentGeneratingIndex != indices.end() ) (*currentGeneratingIndex) = *(currentGeneratingIndex-1);
			}
		}

	}

	// PreCalculator::preCalculate( )
	template< std::size_t D > 
	template<typename real_approx_t> void PreCalculator<D>::preCalculate( Calculator<D,real_approx_t>* calc, const std::array<real_approx_t,D>& t, const real_approx_t p, const real_approx_t lambda ) {

		// // Initialise the data elements. (Probably unnecessary)
		// calc->Q.clear( );
		// calc->dQ_dlambda.clear( );
		// calc->P.clear( );
		// calc->dP_dlambda.clear( );

		calc->slice = 0;


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

		calc->Q.rehash( 3 );
		calc->Q[{0,1}] = -p*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{1,0}] = -p*v[1]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1}] = v[1]*pSq*(-1+v[0])/denominator;

		calc->P.rehash( 4 );
		calc->P[{0,0}] = v[1]*v[0]*(p-1)*(p-1)/denominator;
		calc->P[{0,1}] = -v[1]*v[0]*p*(p-1)/denominator;
		calc->P[{1,0}] = -v[1]*v[0]*p*(p-1)/denominator;
		calc->P[{1,1}] = pSq*v[0]*v[1]/denominator;

		denominator *= denominator;

		calc->dQ_dlambda.rehash( 3 );
		calc->dQ_dlambda[{0,1}] = p*v[1]*(p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]-T[0]*v[0]-T[1]*v[0])/denominator;
		calc->dQ_dlambda[{1,0}] = -p*v[1]*(p-1)*(p*T[0]*v[0]*v[1]-p*T[0]*v[0]-p*T[1]*v[0]-T[0]*v[0]*v[1]+p*T[1])/denominator;
		calc->dQ_dlambda[{1,1}] = v[1]*pSq*(p*T[0]*v[0]*v[1]-p*T[0]*v[0]-p*T[1]*v[0]-T[0]*v[0]*v[1]+p*T[1])/denominator;

		calc->dQ_dlambda.rehash( 4 );
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

		calc->Q.rehash( 7 );
		calc->Q[{0,0,1}] = -p*(-2*p*v[0]*v[1]*v[2]+pp*v[0]*v[1]*v[2]+p*v[1]*v[2]-pp*v[1]*v[2]+v[0]*v[1]*v[2]+p*v[2]-1)/denominator;
		calc->Q[{0,1,0}] = -p*v[2]*(p-1)*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{0,1,1}] = v[2]*pp*(p*v[0]*v[1]-p*v[1]-v[0]*v[1]+1)/denominator;
		calc->Q[{1,0,0}] = -p*v[1]*v[2]*(p-1)*(p-1)*(-1+v[0])/denominator;
		calc->Q[{1,0,1}] = pp*v[1]*v[2]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,0}] = pp*v[1]*v[2]*(-1+v[0])*(p-1)/denominator;
		calc->Q[{1,1,1}] = -v[2]*v[1]*ppp*(-1+v[0])/denominator;

		calc->P.rehash( 8 );
		calc->P[{0,0,0}] = v[0]*v[1]*v[2]*(p-1)*(p-1)*(p-1)/denominator;
		calc->P[{0,0,1}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{0,1,0}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{0,1,1}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,0,0}] = -v[0]*v[1]*v[2]*p*(p-1)*(p-1)/denominator;
		calc->P[{1,0,1}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,1,0}] = v[0]*v[1]*v[2]*pp*(p-1)/denominator;
		calc->P[{1,1,1}] = -ppp*v[0]*v[1]*v[2]/denominator;

		denominator *=  denominator;

		calc->dQ_dlambda.rehash( 7 );
		calc->dQ_dlambda[{0,0,1}] = -p*v[2]*(-2*p*T[0]*v[0]*v[1]-2*p*T[1]*v[0]*v[1]-2*p*T[2]*v[0]*v[1]+pp*T[0]*v[0]*v[1]+pp*T[1]*v[0]*v[1]+pp*T[2]*v[0]*v[1]+p*T[1]*v[1]+p*T[2]*v[1]-pp*T[1]*v[1]-pp*T[2]*v[1]+T[0]*v[0]*v[1]+T[1]*v[0]*v[1]+T[2]*v[0]*v[1]+p*T[2])/denominator;
		calc->dQ_dlambda[{0,1,0}] = p*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]-2*p*T[1]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+pp*T[1]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[1]+p*T[1]*v[0]*v[1]+p*T[1]*v[1]*v[2]+p*T[2]*v[0]*v[1]-pp*T[0]*v[0]*v[1]-pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[0]*v[1]+T[0]*v[0]*v[1]*v[2]+T[1]*v[0]*v[1]*v[2]+pp*T[1]*v[1]+pp*T[2]*v[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{0,1,1}] = -v[2]*pp*(-2*p*T[0]*v[0]*v[1]*v[2]-2*p*T[1]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+pp*T[1]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[1]+p*T[1]*v[0]*v[1]+p*T[1]*v[1]*v[2]+p*T[2]*v[0]*v[1]-pp*T[0]*v[0]*v[1]-pp*T[1]*v[0]*v[1]-pp*T[1]*v[1]*v[2]-pp*T[2]*v[0]*v[1]+T[0]*v[0]*v[1]*v[2]+T[1]*v[0]*v[1]*v[2]+pp*T[1]*v[1]+pp*T[2]*v[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,0,0}] = -p*v[1]*v[2]*(p-1)*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,0,1}] = pp*v[1]*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,1,0}] = pp*v[1]*v[2]*(p-1)*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;
		calc->dQ_dlambda[{1,1,1}] = -v[2]*v[1]*ppp*(-2*p*T[0]*v[0]*v[1]*v[2]+pp*T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]*v[2]+p*T[1]*v[0]*v[2]-pp*T[0]*v[0]*v[2]-pp*T[1]*v[0]*v[2]+T[0]*v[0]*v[1]*v[2]+p*T[0]*v[0]+p*T[1]*v[0]-p*T[1]*v[2]+p*T[2]*v[0]+pp*T[1]*v[2]-p*T[1]-p*T[2])/denominator;

		calc->dP_dlambda.rehash( 8 );
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

		calc->Q.rehash( 15 );
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

		calc->P.rehash( 16 );
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

		calc->dQ_dlambda.rehash( 15 );
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

		calc->dP_dlambda.rehash( 16 );
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

		// Pre calculation
		preCalculate( t, p, lambda );

		// Initialise 
		real_approx_t fisherInformation = static_cast<real_approx_t>(0);
		real_approx_t runningTotal = static_cast<real_approx_t>(-1);

		// Compute the slice delta.
		while( fisherInformation != runningTotal ) {
			runningTotal = fisherInformation;
			fisherInformation = runningTotal + calculateSlice( );
		}

		// Cleanup
		for( auto container : { L, dL_dlambda } ) {
			container.clear();
		}

		for( auto container : { Q, dQ_dlambda, P, dP_dlambda } ) {
			container.clear();
		}

		return fisherInformation;
	}

	// Calculator::initialiseSlice( )
	template< std::size_t D, typename real_approx_t >
	void Calculator<D,real_approx_t>::initialiseSlice( ) {

		// Generate indices for new siice.
		waitingIndices.generate( slice );

		// Clear slice that is no longer needed. (This prevents element copy in the resize, below)
		// NOTE: that both L and dL_dlambda are effectively implemented as circular arrays, so clearing the current slice effectively clears the old unneeded one.
		L[ slice ].clear();
		dL_dlambda[ slice ].clear();

		// Initialise the new slice to the correct size
		L[ slice ].reserve( waitingIndices.size() );
		dL_dlambda[ slice ].reserve( waitingIndices.size() );

		// Increment the slice ready for the next iteration.		
		++slice;
	}

	// Calculator::calculateSlice( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t Calculator<D,real_approx_t>::calculateSlice( )  {

		// Initialise the new slice for computation
		initialiseSlice( );

		// Calculate the Fisher Information delta for this slice by accumulating the delta for each index.
		real_approx_t sliceDelta = static_cast<real_approx_t>( 0 );
		// for( auto idx : waitingIndices ) sliceDelta += calculateIndex( idx );
		for( auto i = 0; i < waitingIndices.size(); ++i ) {
			sliceDelta += calculateIndex( waitingIndices[i] );
		}

		return sliceDelta;
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

	template< std::size_t D, typename real_approx_t >
	void ThreadedCalculator<D,real_approx_t>::preCalculate( const std::array<real_approx_t,D> &t, const real_approx_t p, const real_approx_t lambda ) {
		// Perform parent class precalculation.
		Calculator<D,real_approx_t>::preCalculate( t, p, lambda );

		// Initialise the next slice asynchronously. (Note, parent class initialiseSlice() is called in the absense of a “local” initialiseSlice()).
		// We create the future now, becasue the computeSlice( ) function expects a future to wait on (which is usually launched asynchronously at the end of the previous computeSlice() call).
		sliceInitTask = std::async( std::launch::async, [this]() { ThreadedCalculator<D,real_approx_t>::initialiseSlice(); } );
	}

	// ThreadedCalculator::calculateSlice( )
	template< std::size_t D, typename real_approx_t >
	real_approx_t ThreadedCalculator<D,real_approx_t>::calculateSlice( )  {

		using calc = Calculator<D,real_approx_t>;

		// Initialise the delta value for this slice.
		sliceDelta = static_cast<real_approx_t>(0);

		// Initialise the worker thread completion flags.
		workerThreadCompleteFlags.reset();

		// Wait for initialisation to complete.
		sliceInitTask.wait();

		// Notify workers to start working.
		workerThreadWakeCond.notify_all();

		// Wait for workers to finish.
		{
			std::unique_lock<std::mutex> lck( mtx );
			workerThreadSliceCompletedCond.wait( lck, [this]{ return workerThreadCompleteFlags.all(); } );
		}

		// Initialise the next slice asynchronously. (Note, parent class initialiseSlice() is called in the absense of a “local” initialiseSlice()).
		// NOTE: This actually a detriment when D=2 and lambda is small. This is presumably due to the overhead of calling the thread being higher than the time needed to execute the initialisation.
		// However, it seems to be an improvement in other cases, especially where the runtimes are long, so we keep this method.
		sliceInitTask = std::async( std::launch::async, [this]() { ThreadedCalculator<D,real_approx_t>::initialiseSlice(); } );

		// sliceDelta is automatically accumulated by the indexCalculatorLoop( ) threads.
		return sliceDelta;
	}

	template< std::size_t D, typename real_approx_t >
	void ThreadedCalculator<D,real_approx_t>::indexCalculatorLoop( size_t ID, size_t numThreads ) {

		using calc = Calculator<D,real_approx_t>;

		real_approx_t localTotal;

		std::unique_lock<std::mutex> lck( mtx ); // Lock is locked on construction.
		do {

			// Set the complete flag and notify completion. 
			// Note that we use this mechanism to indicate initial setup completion as well as slice completion
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

			// Calculate the slice stride delta.
			for( auto i = ID; i < calc::waitingIndices.size(); i += numThreads ) {
				localTotal += calc::calculateIndex( calc::waitingIndices[i] );
			}

			// Update the global slice delta.
			lck.lock( );
			sliceDelta += localTotal;

		} while( true );

	}


} // END namespace Fisher

#endif // (FISHER_HH_DECLARATIONS_AND_DEFINITIONS)