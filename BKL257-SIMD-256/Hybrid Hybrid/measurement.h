#define REPEAT 100000
#define REREPEAT 1
#define WARMUP (REPEAT/4)

unsigned long long RDTSC_start_clk, RDTSC_end_clk;
double RDTSC_clk[REPEAT];
double RDTSC_clk_min, RDTSC_clk_median, RDTSC_clk_max;
double min, T1;
int RDTSC_MEASURE_ITERATOR;
int RDTSC_MEASURE_REITERATOR;
int SCHED_RET_VAL;
int i,j,ti;


#define get_Clks(void) ({unsigned long long res; __asm__ __volatile__ ("rdtsc" : "=a"(res) : : "edx"); res;})

#define MEASURE(x) \
	for(RDTSC_MEASURE_ITERATOR=0; RDTSC_MEASURE_ITERATOR< WARMUP; RDTSC_MEASURE_ITERATOR++) \
	{ \
		{x}; \
	}; \
	for (RDTSC_MEASURE_ITERATOR = 0; RDTSC_MEASURE_ITERATOR < REPEAT; RDTSC_MEASURE_ITERATOR++) \
	{ \
		RDTSC_start_clk = get_Clks(); \
		{x}; \
		RDTSC_end_clk = get_Clks(); \
		RDTSC_clk[RDTSC_MEASURE_ITERATOR] = (double)(RDTSC_end_clk-RDTSC_start_clk); \
	}; \
	for (i = 0; i < REPEAT; i++){ \
		min = 	RDTSC_clk[i]; \
		for (j = i+1; j< REPEAT; j++){ \
			if (min > RDTSC_clk[j]){ \
				min = RDTSC_clk[j]; \
				ti = j; \
			} \
		} \
		T1 = RDTSC_clk[ti]; RDTSC_clk[ti] = RDTSC_clk[i]; RDTSC_clk[i] = T1; \
	}; \
	RDTSC_clk_min = RDTSC_clk[0]; \
	RDTSC_clk_median = RDTSC_clk[REPEAT/2]; \
	RDTSC_clk_max = RDTSC_clk[REPEAT-1];
