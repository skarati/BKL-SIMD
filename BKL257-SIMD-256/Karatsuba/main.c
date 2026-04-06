#include <stdio.h>
#include <math.h>

#define align __attribute__ ((aligned (32)))
#if !defined (ALIGN32)
        # if defined (__GNUC__)
                # define ALIGN32 __attribute__ ( (aligned (32)))
        # else
                # define ALIGN32 __declspec (align (32))
        # endif
#endif

#include "BKL257.h"
#include "measurement.h"


int main(){
  
    //scalar Multiplication
    ALIGN32 byte x0[33]={0x08,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; 
    ALIGN32 byte z0[33]={0x01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
                  
    byte n[33] = {0xb4, 0xc9, 0x3c, 0xfd, 0xda, 0x5a, 0xc3, 0x82,
                  0x85, 0xbd, 0x72, 0x17, 0x15, 0xfd, 0x63, 0x3f,
                  0xa3, 0xcc, 0xd9, 0x62, 0xa1, 0x53, 0x29, 0x6e,
                  0xf2, 0xbf, 0x0a, 0xc1, 0xfc, 0x3d, 0x31, 0x05, 0x01};
    
    byte n1[33] = {0xfc, 0xff, 0x3e, 0xfd, 0xda, 0x5a, 0xc3, 0x82,
                   0x80, 0xbd, 0x72, 0x17, 0x15, 0xfd, 0x63, 0x3f,
                   0xa3, 0xcc, 0xd9, 0x62, 0xab, 0xff, 0x29, 0x6e,
                   0xf2, 0xbf, 0x0a, 0xc1, 0xfc, 0x3d, 0x3b, 0xf5, 0x01};
    
    byte n2[33] = {0xfc, 0xff, 0x3e, 0x82, 0xd1, 0x5a, 0xc3, 0x82,
                   0x80, 0xbd, 0x72, 0xf7, 0x05, 0xfd, 0x63, 0x3f,
                   0xa3, 0xcc, 0xd9, 0xff, 0xab, 0xff, 0x29, 0x6e,
                   0xf2, 0xbf, 0x0a, 0xc1, 0xfc, 0x3d, 0x3b, 0xf5, 0x01};
    
                  
    
        
    byte xn[33], zn[33], xnn[33], znn[33];
    
    printf("\n------------------Scalar Multiplication Fixed Base---------------\n");
    
    scalarMult_fixed_base(n, xn, zn);
    
    printf("\n xn = ");
    printBytes(xn,33);
    
    
    
    printf("\n----------------Scalar Multiplication Variable Base--------------\n");
    
    scalarMult_var_base(xn, n, xnn, znn);
    
    printf("\n xnn = ");
    printBytes(xnn,33);
    
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");

    MEASURE({
		scalarMult_fixed_base(n, xn, zn);
	});
	printf("Total CPU cycles for fixed-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");
    
    MEASURE({
		scalarMult_var_base(xn, n, xnn, znn);
	});
	printf("Total CPU cycles for variable-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
    
         
   
    
    return 0;
}
