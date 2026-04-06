#include<immintrin.h>
#include<string.h>


#define byte unsigned char
#define vec __m256i

const vec mask64 = {-1, 0, -1, 0}; 					//LSB_64
const vec mask1 = {1, 0, 1, 0};
const vec mask59 = {0x07FFFFFFFFFFFFFF, 0x0, 0x07FFFFFFFFFFFFFF, 0x0}; //LSB_59
const vec par_b = {1,0,0x86e5,0}; 				       // b | 1 
const vec con[2] = {{0,0,0,0},{-1,-1,-1,-1}};
const vec zero = {0,0,0,0};
const vec baseP[2] = {{1, 0, 0x08, 0}, {0x08, 0, 1, 0}};
const vec _2baseP  = {0x40, 0, 0x86ed6e5, 0};


#define vadd(C,A,B) {C = _mm256_xor_si256(A, B);}
#define vsub(C,A,B) {C = _mm256_sub_epi64(A, B);}
#define vmult(C,A,B) {C = _mm256_clmulepi64_epi128(A, B, 0x00);}
#define vshiftl64(C,A,B) {C=_mm256_slli_epi64(A,B);}
#define vshiftr(C,A,B) {C=_mm256_srli_si256(A,B);}
#define vshiftr64(C,A,B) {C=_mm256_srli_epi64(A,B);}
#define vand(C,A,B) {C=_mm256_and_si256(A,B);}


//Function to print a array consisting some specific number of bytes 
void printBytes(byte *data, int num){
    for (int i=num-1; i>=0; i--)
        printf("%02x",data[i]);
    printf("\n");
}    

//Function to pack two different numbers src1, src2 as (src1 | src2) into four 256 bits registers
void pack64(byte *src1, byte *src2, __m256i *dest){
    __m128i dest1[5], dest2[5];
    byte temp[8] = {0};
    
    temp[0] = src1[32];
    //Storing 'src1' in lower 64 parts of four 128 bits registers
    dest1[0] = _mm_loadu_si128((__m128i*)src1);
    dest1[1] = _mm_loadu_si128((__m128i*)(src1 + 8));
    dest1[2] = _mm_loadu_si128((__m128i*)(src1 + 16));
    dest1[3] = _mm_loadu_si128((__m128i*)(src1 + 24));
    dest1[4] = _mm_loadu_si128((__m128i*)(temp));

    temp[0] = src2[32];
    //Storing 'src2' in lower 64 parts of four 128 bits registers
    dest2[0] = _mm_loadu_si128((__m128i*)src2);
    dest2[1] = _mm_loadu_si128((__m128i*)(src2 + 8));
    dest2[2] = _mm_loadu_si128((__m128i*)(src2 + 16));
    dest2[3] = _mm_loadu_si128((__m128i*)(src2 + 24));
    dest2[4] = _mm_loadu_si128((__m128i*)(temp));


    //parallel alignment of src1 and src2 in 256 bits register respectively
    dest[0] = _mm256_set_m128i(dest2[0], dest1[0]);	vand(dest[0],dest[0],mask64);
    dest[1] = _mm256_set_m128i(dest2[1], dest1[1]); 	vand(dest[1],dest[1],mask64);
    dest[2] = _mm256_set_m128i(dest2[2], dest1[2]);	vand(dest[2],dest[2],mask64);
    dest[3] = _mm256_set_m128i(dest2[3], dest1[3]);	vand(dest[3],dest[3],mask64);
    dest[4] = _mm256_set_m128i(dest2[4], dest1[4]);	vand(dest[4],dest[4],mask64);
}

void unpack64(__m256i *src, byte *dest1, byte *dest2){
    // Store the 256-bit register into a temporary array
    int	i;
    byte temp[32];
    for(i = 0; i < 4; i++){
        _mm256_storeu_si256((__m256i*)temp, src[i]);
        // Extract the lower 64 bits (first 8 bytes) and third 64 bits (third 8 bytes)
    	memcpy(dest1 + i*8, temp, 8);        // Lower 64 bits are at offset 0
    	memcpy(dest2 + i*8, temp + 16, 8);   // Third 64 bits are at offset 16
    } 
    _mm256_storeu_si256((__m256i*)temp, src[4]);
    // Extract the lower 64 bits (first 8 bytes) and third 64 bits (third 8 bytes)
    memcpy(dest1 + i*8, temp, 1);        // Lower 64 bits are at offset 0
    memcpy(dest2 + i*8, temp + 16, 1);   // Third 64 bits are at offset 16
}


// Schoolbook for BKL-251
#define vmult4(u0,u1,u2,u3,v0,v1,v2,v3,w0,w1,w2,w3,w4,w5,w6){ \
	vmult(w0, u0, v0); \
	vmult(t0, u0, v1); \
	vmult(t1, u1, v0); \
	vadd(w1, t0, t1); \
	\
	vmult(w2, u2, v0); \
	vmult(t1, u0, v2); \
	vmult(t2, u1, v1); \
	vadd(w2, w2, t1); \
	vadd(w2, w2, t2);\
	\
	vmult(w3, u3, v0); \
	vmult(t1, u0, v3); \
	vmult(t2, u2, v1); \
	vmult(t3, u1, v2); \
	vadd(t4, w3, t1); \
	vadd(t5, t4, t2); \
	vadd(w3, t5, t3); \
	\
	vmult(w4, u3, v1); \
	vmult(t3, u1, v3); \
	vmult(t4, u2, v2); \
	vadd(t5, w4, t3); \
	vadd(w4, t5, t4); \
	\
	vmult(t6, u3, v2); \
	vmult(t7, u2, v3); \
	vadd(w5, t6, t7); \
	\
	vmult(w6, u3, v3); \
}


vec t00, t01, t10, t11, t02, t20, t03, t30, t12, t21, t04, t40, t22, t13, t31, t23, t32, t14, t41, t24, t42, t33, t34, t43, t44;
vec t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, c1, c2;


// Hybrid Schoolbook Multiplication with 18 vpclmulqdq and 9 vpand
#define vmult5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
	vmult4(u0,u1,u2,u3,v0,v1,v2,v3,w0,w1,w2,w3,w4,w5,w6); \
	vsub(c1, zero, u4); \
	vsub(c2, zero, v4); \
	vand(t40, c1, v0); \
	\
	vand(t04, u0, c2); \
	vadd(t6, t04, t40); \
	vadd(w4, w4, t6); \
	vand(t14, u1, c2); \
	\
	vand(t41, c1, v1); \
	vadd(t0, t14, t41); \
	vadd(w5, w5, t0); \
	vand(t24, u2, c2); \
	\
	vand(t42, c1, v2); \
	vadd(t7, t24, t42); \
	vadd(w6, w6, t7); \
	vand(t34, u3, c2); \
	\
	vand(t43, c1, v3); \
	vadd(w7, t34, t43); \
	vand(w8, u4, v4); \
}




#define vsq5(u0,u1,u2,u3,u4,w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
	vmult(w0, u0, u0); \
	vmult(w2, u1, u1); \
	vmult(w4, u2, u2); \
	vmult(w6, u3, u3); \
	vand(w8, u4, u4); \
}


#define vmulC(u0,u1,u2,u3,u4,c,wf0,wf1,wf2,wf3,wf4){ \
	vmult(wf0, u0, c); \
	vmult(wf1, u1, c); \
	vmult(wf2, u2, c); \
	vmult(wf3, u3, c); \
	vmult(wf4, u4, c); \
}

#define add5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
	vadd(w4, u4, v4); \
}

#define and5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4){ \
	vand(w0, u0, v0); \
	vand(w1, u1, v1); \
	vand(w2, u2, v2); \
	vand(w3, u3, v3); \
	vand(w4, u4, v4); \
}

#define add9(u0,u1,u2,u3,u4,u5,u6,u7,u8,v0,v1,v2,v3,v4,v5,v6,v7,v8,w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
	vadd(w4, u4, v4); \
	vadd(w5, u5, v5); \
	vadd(w6, u6, v6); \
	vadd(w7, u7, v7); \
	vadd(w8, u8, v8); \
}


#define expandM(w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
    	vshiftr(t1, w0, 8); vshiftr(t2, w1, 8); vshiftr(t3, w2, 8); vshiftr(t4, w3, 8); \
    	vshiftr(t5, w4, 8); vshiftr(t6, w5, 8); vshiftr(t7, w6, 8); \
    	\
    	vadd(w1, w1, t1); vadd(w2, w2, t2); vadd(w3, w3, t3); vadd(w4, w4, t4); \
    	vadd(w5, w5, t5); vadd(w6, w6, t6); vadd(w7, w7, t7); \
    	\
    	vand(w0, w0, mask64); vand(w1, w1, mask64); vand(w2, w2, mask64); vand(w3, w3, mask64); \
    	vand(w4, w4, mask64); vand(w5, w5, mask64); vand(w6, w6, mask64); vand(w7, w7, mask64); \
}


#define expandS(w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
    	vshiftr(w1, w0, 8); vshiftr(w3, w2, 8); vshiftr(w5, w4, 8); vshiftr(w7, w6, 8); \
    	vand(w0, w0, mask64); vand(w2, w2, mask64); vand(w4, w4, mask64); vand(w6, w6, mask64);\
}

vec tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
vec tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
vec a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, b0;

#define foldM(w0,w1,w2,w3,w4,w5,w6,w7,w8,wf0,wf1,wf2,wf3,wf4){\
        vshiftl64(tmp0, w5, 63); vshiftl64(tmp1, w5, 11); vshiftr64(tmp2, w5, 1); vshiftr64(tmp3, w5, 53); \
        vshiftl64(tmp4, w6, 63); vshiftl64(tmp5, w6, 11); vshiftr64(tmp6, w6, 1); vshiftr64(tmp7, w6, 53); \
        vshiftl64(tmp8, w7, 63); vshiftl64(tmp9, w7, 11); vshiftr64(tmp10, w7, 1); vshiftr64(tmp11, w7, 53); \
        vshiftl64(tmp12, w8, 63); vshiftl64(tmp13, w8, 11); vshiftr64(tmp14, w8, 1); \
        \
        vadd(wf0, w0, tmp0); vadd(a0, w1, tmp1); vadd(a1, a0, tmp2); vadd(wf1, a1, tmp4); \
        vadd(a2, w2, tmp3); vadd(a3, a2, tmp5); vadd(a4, a3, tmp6); vadd(wf2, a4, tmp8); \
        vadd(a5, w3, tmp7); vadd(a6, a5, tmp9); vadd(a7, a6, tmp10); vadd(wf3, a7, tmp12); \
        vadd(a8, w4, tmp11); vadd(a9, a8, tmp13); vadd(wf4, a9, tmp14); \
} 

#define foldS(w0,w1,w2,w3,w4,w5,w6,w7,w8,wf0,wf1,wf2,wf3,wf4){\
        vshiftl64(tmp0, w5, 63); vshiftl64(tmp1, w5, 11); vshiftr64(tmp2, w5, 1); vshiftr64(tmp3, w5, 53); \
        vshiftl64(tmp4, w6, 63); vshiftl64(tmp5, w6, 11); vshiftr64(tmp6, w6, 1); vshiftr64(tmp7, w6, 53); \
        vshiftl64(tmp8, w7, 63); vshiftl64(tmp9, w7, 11); vshiftr64(tmp10, w7, 1); vshiftr64(tmp11, w7, 53); \
        vshiftl64(tmp12, w8, 63); vshiftl64(tmp13, w8, 11); \
        \
        vadd(wf0, w0, tmp0); vadd(a0, w1, tmp1); vadd(a1, a0, tmp2); vadd(wf1, a1, tmp4); \
        vadd(a2, w2, tmp3); vadd(a3, a2, tmp5); vadd(a4, a3, tmp6); vadd(wf2, a4, tmp8); \
        vadd(a5, w3, tmp7); vadd(a6, a5, tmp9); vadd(a7, a6, tmp10); vadd(wf3, a7, tmp12); \
        vadd(a8, w4, tmp11); vadd(wf4, a8, tmp13); \
}

#define reduce(wf0,wf1,wf2,wf3,wf4){\
        vshiftr64(tmp15, wf4, 1); vand(wf4, wf4, mask1); vshiftl64(tmp16, tmp15, 12); vshiftr64(tmp17, tmp15, 52); \
        vadd(b0, wf0, tmp15); vadd(wf0, b0, tmp16); vadd(wf1, wf1, tmp17); \
}



#define reduceC(u0,u1,u2,u3,u4,wf0,wf1,wf2,wf3,wf4){\
        vand(t0, u0, mask64); \
        vand(t1, u1, mask64); \
        vand(t2, u2, mask64); \
        vand(t3, u3, mask64); \
        \
        vshiftr(tmp0, u0, 8); \
        vshiftr(tmp1, u1, 8); \
        vshiftr(tmp2, u2, 8); \
        vshiftr(tmp3, u3, 8); \
        \
        vadd(wf1, t1, tmp0); \
        vadd(wf2, t2, tmp1); \
        vadd(wf3, t3, tmp2); \
        vadd(u4, u4, tmp3); \
        \
        vshiftr64(t4, u4, 1); \
        vand(wf4, u4, mask1); \
        vshiftl64(t5, t4, 12); \
        vshiftr64(t6, t4, 52); \
        \
        vadd(t7, t4, t5); \
        vadd(wf0, t0, t7); \
        vadd(wf1, wf1, t6); \
}


#define swap(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,B){ \
	vadd(t0, u0, v0); vadd(t1, u1, v1); vadd(t2, u2, v2); vadd(t3, u3, v3); vadd(t4, u4, v4); \
	vand(t0, t0, B); vand(t1, t1, B); vand(t2, t2, B); vand(t3, t3, B); vand(t4, t4, B); \
	vadd(u0, u0, t0); vadd(u1, u1, t1); vadd(u2, u2, t2); vadd(u3, u3, t3); vadd(u4, u4, t4); \
	vadd(v0, v0, t0); vadd(v1, v1, t1); vadd(v2, v2, t2); vadd(v3, v3, t3); vadd(v4, v4, t4); \
}


// Ladder-Step Fixed Base
void ladderStepF(vec *x1z1, vec *x2z2, vec zx){
       
	vec temp1[5], temp2[5], temp3[5], A[9], B[9], E[5], E2[5], res[5];
	vec B2[5], C[5], F[5], G[5], G2[5];
	vec w[9],wf[5];
   	
	// temp1 = x1 | x1                                              
	temp1[0] = _mm256_permute2x128_si256(x1z1[0], x1z1[0], 0x11);   
	temp1[1] = _mm256_permute2x128_si256(x1z1[1], x1z1[1], 0x11);   
	temp1[2] = _mm256_permute2x128_si256(x1z1[2], x1z1[2], 0x11);   
	temp1[3] = _mm256_permute2x128_si256(x1z1[3], x1z1[3], 0x11);   
	temp1[4] = _mm256_permute2x128_si256(x1z1[4], x1z1[4], 0x11);   
	        
		
	// temp2 = z2 | x2	
	temp2[0] = _mm256_permute2x128_si256(x2z2[0], x2z2[0], 0x01);
	temp2[1] = _mm256_permute2x128_si256(x2z2[1], x2z2[1], 0x01);
	temp2[2] = _mm256_permute2x128_si256(x2z2[2], x2z2[2], 0x01);
	temp2[3] = _mm256_permute2x128_si256(x2z2[3], x2z2[3], 0x01);
	temp2[4] = _mm256_permute2x128_si256(x2z2[4], x2z2[4], 0x01);
	
	
	// temp3 = z1 | z1
	temp3[0] = _mm256_permute2x128_si256(x1z1[0], x1z1[0], 0x00);
	temp3[1] = _mm256_permute2x128_si256(x1z1[1], x1z1[1], 0x00);
	temp3[2] = _mm256_permute2x128_si256(x1z1[2], x1z1[2], 0x00);
	temp3[3] = _mm256_permute2x128_si256(x1z1[3], x1z1[3], 0x00);
	temp3[4] = _mm256_permute2x128_si256(x1z1[4], x1z1[4], 0x00);
	

	//B2 = z1 | 0
        B2[0] = _mm256_permute2x128_si256(x1z1[0], zero, 0x02);
	B2[1] = _mm256_permute2x128_si256(x1z1[1], zero, 0x02);
	B2[2] = _mm256_permute2x128_si256(x1z1[2], zero, 0x02);
	B2[3] = _mm256_permute2x128_si256(x1z1[3], zero, 0x02);
	B2[4] = _mm256_permute2x128_si256(x1z1[4], zero, 0x02);
	
	

    	
        // A = x1x2 | z2x1
        vmult5(x2z2[0],x2z2[1],x2z2[2],x2z2[3],x2z2[4],
    	       temp1[0],temp1[1],temp1[2],temp1[3],temp1[4],
    	       A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8]);
    	  
    
    	
    	
    	// B = z1z2 | x2z1
    	vmult5(temp2[0],temp2[1],temp2[2],temp2[3],temp2[4],
    	       temp3[0],temp3[1],temp3[2],temp3[3],temp3[4],
    	       B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8]);
    	       
    	       
    	// E = x1x2 + z1z2 | x2z1 + x1z2
    	add9(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8],
    	     B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8],
    	     w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	     
    	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],
    	        w[4],w[5],w[6],w[7],w[8],
    	        E[0],E[1],E[2],E[3],E[4]);
    	        
    	reduce(E[0],E[1],E[2],E[3],E[4]);
    	

    	
    	
    	// E2 = (x1x2 + z1z2)^2 | (x2z1 + x1z2)^2
    	vsq5(E[0],E[1],E[2],E[3],E[4],
    	     w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],
    	      w[4],w[5],w[6],w[7],w[8],
    	      E2[0],E2[1],E2[2],E2[3],E2[4]);
    	      
    	reduce(E2[0],E2[1],E2[2],E2[3],E2[4]);
    	
    	
    	
    	
    	// x4z4 =  z(x1x2 + z1z2)^2 | x(x2z1 + x1z2)^2
	vmulC(E2[0],E2[1],E2[2],E2[3],E2[4],
	      zx,wf[0],wf[1],wf[2],wf[3],wf[4]);
        
        reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
                x2z2[0],x2z2[1],x2z2[2],x2z2[3],x2z2[4]);
    	
    	
	
	// E = x1 + z1 | z1
    	add5(x1z1[0],x1z1[1],x1z1[2],x1z1[3],x1z1[4],
    	     B2[0],B2[1],B2[2],B2[3],B2[4],
    	     E[0],E[1],E[2],E[3],E[4]);
    	
    	// F = z1 + x1 | x1
    	add5(B2[0],B2[1],B2[2],B2[3],
    	     B2[4],temp1[0],temp1[1],temp1[2],
    	     temp1[3],temp1[4],F[0],F[1],F[2],F[3],F[4]);


    	// G = x1^2 + z1^2 | z1*x1
    	vmult5(E[0],E[1],E[2],E[3],E[4],
    	       F[0],F[1],F[2],F[3],F[4],
    	       w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	       
    	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	      G[0],G[1],G[2],G[3],G[4]);
    	      
    	reduce(G[0],G[1],G[2],G[3],G[4]);
    	
    	
    	      
    	// G2 = (x1^2 + z1^2)^2 | (x1z1)^2
    	vsq5(G[0],G[1],G[2],G[3],G[4],
    	       w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	       
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	        G2[0],G2[1],G2[2],G2[3],G2[4]);
    	        
    	reduce(G2[0],G2[1],G2[2],G2[3],G2[4]);
    	
    	
    
    	
    	// res = b*(x1^2 + z1^2)^2 | (x1z1)^2
	vmulC(G2[0],G2[1],G2[2],G2[3],G2[4],
	      par_b,wf[0],wf[1],wf[2],wf[3],wf[4]);
        
        reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
                x1z1[0],x1z1[1],x1z1[2],x1z1[3],x1z1[4]);
        
                
}



// Ladder Step Variable Base

void ladderStepV(vec *x1z1, vec *x2z2, vec *vb_zx){
    
    vec temp1[5], temp2[5], temp3[5], A[9], B[9], E[5], E2[5], res[5];
    vec B2[5], C[5], F[5], G[5], G2[5];
    vec w[9],wf[5];
   
    
    // temp1 = x1 | x1                                              
    temp1[0] = _mm256_permute2x128_si256(x1z1[0], x1z1[0], 0x11);   
    temp1[1] = _mm256_permute2x128_si256(x1z1[1], x1z1[1], 0x11);   
    temp1[2] = _mm256_permute2x128_si256(x1z1[2], x1z1[2], 0x11);   
    temp1[3] = _mm256_permute2x128_si256(x1z1[3], x1z1[3], 0x11);   
    temp1[4] = _mm256_permute2x128_si256(x1z1[4], x1z1[4], 0x11);   
	        
		
    // temp2 = z2 | x2	
    temp2[0] = _mm256_permute2x128_si256(x2z2[0], x2z2[0], 0x01);
    temp2[1] = _mm256_permute2x128_si256(x2z2[1], x2z2[1], 0x01);
    temp2[2] = _mm256_permute2x128_si256(x2z2[2], x2z2[2], 0x01);
    temp2[3] = _mm256_permute2x128_si256(x2z2[3], x2z2[3], 0x01);
    temp2[4] = _mm256_permute2x128_si256(x2z2[4], x2z2[4], 0x01);
    
    	
    // temp3 = z1 | z1
    temp3[0] = _mm256_permute2x128_si256(x1z1[0], x1z1[0], 0x00);
    temp3[1] = _mm256_permute2x128_si256(x1z1[1], x1z1[1], 0x00);
    temp3[2] = _mm256_permute2x128_si256(x1z1[2], x1z1[2], 0x00);
    temp3[3] = _mm256_permute2x128_si256(x1z1[3], x1z1[3], 0x00);
    temp3[4] = _mm256_permute2x128_si256(x1z1[4], x1z1[4], 0x00);
    
    //B2 = z1 | 0
    B2[0] = _mm256_permute2x128_si256(x1z1[0], zero, 0x02);
    B2[1] = _mm256_permute2x128_si256(x1z1[1], zero, 0x02);
    B2[2] = _mm256_permute2x128_si256(x1z1[2], zero, 0x02);
    B2[3] = _mm256_permute2x128_si256(x1z1[3], zero, 0x02);
    B2[4] = _mm256_permute2x128_si256(x1z1[4], zero, 0x02);
    

    	
    // A = x1x2 | z2x1
    vmult5(x2z2[0],x2z2[1],x2z2[2],x2z2[3],x2z2[4],
    	  temp1[0],temp1[1],temp1[2],temp1[3],temp1[4],
    	  A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8]);
    	
    	
    // B = z1z2 | x2z1
    vmult5(temp2[0],temp2[1],temp2[2],temp2[3],temp2[4],
    	   temp3[0],temp3[1],temp3[2],temp3[3],temp3[4],
    	   B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8]);
    	       
    	       
    // E = x1x2 + z1z2 | x2z1 + x1z2
    add9(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8],
    	 B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8],
    	 w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	 
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
            E[0],E[1],E[2],E[3],E[4]);
    
    reduce(E[0],E[1],E[2],E[3],E[4]);    
    
    // E2 = (x1x2 + z1z2)^2 | (x2z1 + x1z2)^2               
    vsq5(E[0],E[1],E[2],E[3],E[4],
    	 w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	 
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
            E2[0],E2[1],E2[2],E2[3],E2[4]);
    
    reduce(E2[0],E2[1],E2[2],E2[3],E2[4]);
    	
    // x4z4 =  z(x1x2 + z1z2)^2 | x(x2z1 + x1z2)^2
    vmult5(E2[0],E2[1],E2[2],E2[3],E2[4],
	   vb_zx[0],vb_zx[1],vb_zx[2],vb_zx[3],vb_zx[4],
	   w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	   
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
            x2z2[0],x2z2[1],x2z2[2],x2z2[3],x2z2[4]);
    
    reduce(x2z2[0],x2z2[1],x2z2[2],x2z2[3],x2z2[4]);

    	
    	
	
    // E = x1 + z1 | z1
    add5(x1z1[0],x1z1[1],x1z1[2],x1z1[3],x1z1[4],
    	 B2[0],B2[1],B2[2],B2[3],B2[4],
    	 E[0],E[1],E[2],E[3],E[4]);
    	
    // F = z1 + x1 | x1
    add5(temp1[0],temp1[1],temp1[2],temp1[3],temp1[4],
         B2[0],B2[1],B2[2],B2[3],B2[4],
    	 F[0],F[1],F[2],F[3],F[4]);


    // G = x1^2 + z1^2 | z1*x1
    vmult5(E[0],E[1],E[2],E[3],E[4],
    	   F[0],F[1],F[2],F[3],F[4],
    	   w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	   
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
            G[0],G[1],G[2],G[3],G[4]);
    
    reduce(G[0],G[1],G[2],G[3],G[4]);
    	      
    // G2 = (x1^2 + z1^2)^2 | (x1z1)^2
    vsq5(G[0],G[1],G[2],G[3],G[4],
    	 w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	 
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          G2[0],G2[1],G2[2],G2[3],G2[4]);
            
    reduce(G2[0],G2[1],G2[2],G2[3],G2[4]);
    	
    // res = b*(x1^2 + z1^2)^2 | (x1z1)^2
    vmulC(G2[0],G2[1],G2[2],G2[3],G2[4],
          par_b,wf[0],wf[1],wf[2],wf[3],wf[4]);
    
    reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
            x1z1[0],x1z1[1],x1z1[2],x1z1[3],x1z1[4]);
        
}


// Function for field inversion
void invert(vec *in, vec *op){
    vec t[5];
    vec x2[5],x3[5],x4[5],x7[5],x_6_1[5],x_12_1[5],x_24_1[5],x_25_1[5],x_50_1[5],x_100_1[5],x_125_1[5],x_250_1[5],x_256_1[5];
    vec temp[5];
    
    vec w[9],wf[5];
    
    // 2
    vsq5(in[0], in[1], in[2], in[3],in[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x2[0],x2[1],x2[2],x2[3],x2[4]);
    reduce(x2[0],x2[1],x2[2],x2[3],x2[4]);
    
    // 3
    vmult5(in[0],in[1],in[2],in[3],in[4], 
           x2[0],x2[1],x2[2],x2[3],x2[4],
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x3[0],x3[1],x3[2],x3[3],x3[4]);
    
    reduce(x3[0],x3[1],x3[2],x3[3],x3[4]);
    
    // 4
    vsq5(x2[0],x2[1],x2[2],x2[3],x2[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x4[0],x4[1],x4[2],x4[3],x4[4]);
    
    reduce(x4[0],x4[1],x4[2],x4[3],x4[4]);
    
    // 7
    vmult5(x3[0],x3[1],x3[2],x3[3],x3[4], 
           x4[0],x4[1],x4[2],x4[3],x4[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x7[0],x7[1],x7[2],x7[3],x7[4]);
    
    reduce(x7[0],x7[1],x7[2],x7[3],x7[4]);
    
    for(int i=0; i<5;i++){temp[i] = x7[i];}
    
    // 2^6-8
    for(int i=0;i<3;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    
    // 2^6-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x7[0],x7[1],x7[2],x7[3],x7[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4]);
    
    reduce(x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_6_1[i];}
    // 2^12-2^6
    for(int i=0;i<6;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    // 2^12-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4]);
    
    reduce(x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_12_1[i];}
    //2^24-2^12
    for(int i=0;i<12;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^24-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_24_1[0],x_24_1[1],x_24_1[2],x_24_1[3],x_24_1[4]);
    
    reduce(x_24_1[0],x_24_1[1],x_24_1[2],x_24_1[3],x_24_1[4]);
    
    //2^25-2
    vsq5(x_24_1[0], x_24_1[1], x_24_1[2], x_24_1[3],x_24_1[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    reduce(x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    //2^25-1
    vmult5(in[0],in[1],in[2],in[3],in[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    reduce(x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_25_1[i];}
    //2^50-2^25
    for(int i=0;i<25;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^50-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4]);
    
    reduce(x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_50_1[i];}
    //2^100-2^50
    for(int i=0;i<50;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    // 2^100-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_100_1[0],x_100_1[1],x_100_1[2],x_100_1[3],x_100_1[4]);
    
    reduce(x_100_1[0],x_100_1[1],x_100_1[2],x_100_1[3],x_100_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_100_1[i];}
    // 2^125-2^25
    for(int i=0;i<25;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^125-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4]);
    
    reduce(x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_125_1[i];}
    //2^250-2^125
    for(int i=0;i<125;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^250-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3],x_250_1[4]);
    
    reduce(x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3],x_250_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_250_1[i];}
    // 2^256 - 2^6
    for(int i=0;i<6;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    
    // 2^256 -1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4]);
    
    reduce(x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4]);
    
    // 2^257 -2
    vsq5(x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], op[0],op[1],op[2],op[3],op[4]);
    
    reduce(op[0],op[1],op[2],op[3],op[4]);    
    
}


//Function to clamp the scalar
void clamp(byte n[33]){
	n[0] = n[0] & 0xfc;
	
	n[32] = n[32] | 0x01;
	
}

// Fixed Base Scalar Multiplication
void scalarMult_fixed_base(byte *n, byte *op_x, byte *op_z ){  
    
    clamp(n);
    
    vec S[5], R[5];
    vec nP[5];
    vec w[9],wf[5];
    vec zx = baseP[1];
   
    
    S[0] = baseP[0];   R[0] = _2baseP;
    S[1] = zero;         R[1] = zero;
    S[2] = zero;         R[2] = zero;
    S[3] = zero;         R[3] = zero;
    S[4] = zero;         R[4] = zero;
                             

    byte pb = 0, b, ni;
     
    int j;
    for(int i = 31; i >= 0 ; i--){
        j = 7;
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
          
            swap(S[0],S[1],S[2],S[3],S[4],
                 R[0],R[1],R[2],R[3],R[4],con[b]);
                   
            ladderStepF(S, R, zx);
     
            pb = ni;
        }
        
        
    }
    
    swap(S[0],S[1],S[2],S[3],S[4],
         R[0],R[1],R[2],R[3],R[4],con[pb]);
 
    
    vec X[5],Z[5];
         
    invert(S,Z);
    
    X[0] = _mm256_permute2x128_si256(S[0], zero, 0x11);
    X[1] = _mm256_permute2x128_si256(S[1], zero, 0x11);
    X[2] = _mm256_permute2x128_si256(S[2], zero, 0x11);
    X[3] = _mm256_permute2x128_si256(S[3], zero, 0x11);
    X[4] = _mm256_permute2x128_si256(S[4], zero, 0x11);
    
    vmult5(X[0],X[1],X[2],X[3],X[4],
	   Z[0],Z[1],Z[2],Z[3],Z[4],
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          nP[0],nP[1],nP[2],nP[3],nP[4]);
          
    reduce(nP[0],nP[1],nP[2],nP[3],nP[4]);
                  

    unpack64(nP, op_x, op_z);
    
} 


//variable Base Scalar Multiplication

void scalarMult_var_base( byte *x, byte *n, byte *nx, byte *nz ){  
    
    
    clamp(n);
    
    vec Pxz[5], nP[5], S[5], R[5];
    vec w[9],wf[5];
    byte z1[33] = {0x01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    pack64(z1, x, S); //S = x | z
    vec vb_zx[5];
    
    vb_zx[0] = _mm256_permute2x128_si256(S[0], S[0], 0x01);
    vb_zx[1] = _mm256_permute2x128_si256(S[1], S[1], 0x01);
    vb_zx[2] = _mm256_permute2x128_si256(S[2], S[2], 0x01);
    vb_zx[3] = _mm256_permute2x128_si256(S[3], S[3], 0x01);
    vb_zx[4] = _mm256_permute2x128_si256(S[4], S[4], 0x01);
    
    
     
    
    //S = x | z
    vec B2[5], E[5], F[5], G[5], G2[5], temp1[5], L[5], M[5], N[5], B3[5], O[5];
        
    byte pb = 0, b, ni;
    
    //B2 = z | 0
    B2[0] = _mm256_permute2x128_si256(S[0], zero, 0x02);
    B2[1] = _mm256_permute2x128_si256(S[1], zero, 0x02);
    B2[2] = _mm256_permute2x128_si256(S[2], zero, 0x02);
    B2[3] = _mm256_permute2x128_si256(S[3], zero, 0x02);
    B2[4] = _mm256_permute2x128_si256(S[4], zero, 0x02);
    
	
    // E = x + z | z
    add5(B2[0],B2[1],B2[2],B2[3],B2[4],
         S[0],S[1],S[2],S[3],S[4],
    	 E[0],E[1],E[2],E[3],E[4]);
    	
          
    	
    // F = z + x | x
    
    F[0] = _mm256_permute2x128_si256(S[0], E[0], 0x31);
    F[1] = _mm256_permute2x128_si256(S[1], E[1], 0x31);
    F[2] = _mm256_permute2x128_si256(S[2], E[2], 0x31);
    F[3] = _mm256_permute2x128_si256(S[3], E[3], 0x31);
    F[4] = _mm256_permute2x128_si256(S[4], E[4], 0x31);
         

    
    // G = x^2 + z^2 | z*x
    vmult5(E[0],E[1],E[2],E[3],E[4],
           F[0],F[1],F[2],F[3],F[4],
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
          
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]); 
   
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          G[0],G[1],G[2],G[3],G[4]);
          
    reduce(G[0],G[1],G[2],G[3],G[4]);
            

                	      
    // G2 = (x^2 + z^2)^2 | (xz)^2
    vsq5(G[0],G[1],G[2],G[3],G[4],
         w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
         
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          G2[0],G2[1],G2[2],G2[3],G2[4]);
          
    reduce(G2[0],G2[1],G2[2],G2[3],G2[4]);
            
       	
    // res = b*(x^2 + z^2)^2 | (xz)^2
    vmulC(G2[0],G2[1],G2[2],G2[3],G2[4],
          par_b,wf[0],wf[1],wf[2],wf[3],wf[4]);
          
    reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
            R[0],R[1],R[2],R[3],R[4]);
    
      
    int j;
    for(int i = 31; i >= 0 ; i--){
        j = 7;
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
          
            swap(S[0],S[1],S[2],S[3],S[4],
                 R[0],R[1],R[2],R[3],R[4],con[b]);
            
            ladderStepV(S, R, vb_zx);
            
            pb = ni;
        }
        
    }
    
    swap(S[0],S[1],S[2],S[3],S[4],
         R[0],R[1],R[2],R[3],R[4],con[pb]);
    
    
    vec X[5],Z[5];
         
    invert(S,Z);
    
    X[0] = _mm256_permute2x128_si256(S[0], zero, 0x11);
    X[1] = _mm256_permute2x128_si256(S[1], zero, 0x11);
    X[2] = _mm256_permute2x128_si256(S[2], zero, 0x11);
    X[3] = _mm256_permute2x128_si256(S[3], zero, 0x11);
    X[4] = _mm256_permute2x128_si256(S[4], zero, 0x11);
    
    vmult5(X[0],X[1],X[2],X[3],X[4], 
           Z[0],Z[1],Z[2],Z[3],Z[4], 
	   w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	   
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          nP[0],nP[1],nP[2],nP[3],nP[4]);
          
    reduce(nP[0],nP[1],nP[2],nP[3],nP[4]);
                  

    unpack64(nP, nx, nz);
    
}
