#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define SHARED_SUBGROUP __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define SHARED_SUBGROUP
#endif

#ifdef SHARED_DP4A
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
inline int shared_dot8(uint8 w, __local const uint * a) {
    int s=0;
    s=dot_acc_sat_4x8packed_ss_int(w.s0,a[0],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s1,a[1],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s2,a[2],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s3,a[3],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s4,a[4],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s5,a[5],s);
    s=dot_acc_sat_4x8packed_ss_int(w.s6,a[6],s);
    return dot_acc_sat_4x8packed_ss_int(w.s7,a[7],s);
}

// Same 64-row x 32-token integer dot tile as the native dense path.
// CPU4x8 quants are read directly; only activations use transient Q8 storage.
__attribute__((qcom_wave_pair_mode(1)))
__kernel void shared_q4_0_dp4a(__global const uint * weights, ulong wo,
        __global const uint * qa, __global const half * da, __global const half * sa,
        __global uchar * output, ulong yo, uint m, uint n, uint k) {
    uint lane=get_local_id(0),row=get_global_id(1)*64+lane;
    uint token=get_global_id(2)*32,rr=min(row,m-1),kb=k/32;
    __global float * y=(__global float *)(output+yo);
    __local uint aq[32][8];
    __local half ad[32],asum[32];
    float4 sums[8];
    #pragma unroll
    for(uint g=0;g<8;++g) sums[g]=0;
    for(uint b=0;b<kb;++b) {
        uint base=(uint)(wo/4)+((rr/4)*kb+b)*18;
        float scale=(float)((__global const half *)(weights+base))[rr%4];
        uint2 lo=vload2(0,weights+base+2+(rr%4)*2)^0x88888888u;
        uint2 hi=vload2(0,weights+base+10+(rr%4)*2)^0x88888888u;
        uint4 bytes=(uint4)(lo,hi);
        uint8 w=(uint8)(bytes&0x0f0f0f0fu,(bytes>>4)&0x0f0f0f0fu);
        for(uint i=lane;i<256;i+=64) {
            uint t=i/8,u=i%8;
            aq[t][u]=token+t<n?qa[(token+t)*(k/4)+b*8+u]:0;
        }
        if(lane<32) {
            ad[lane]=token+lane<n?da[(token+lane)*kb+b]:(half)0;
            asum[lane]=token+lane<n?sa[(token+lane)*kb+b]:(half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(uint g=0;g<8;++g) {
            uint t=g*4;
            float4 dot=(float4)(shared_dot8(w,aq[t]),shared_dot8(w,aq[t+1]),
                                shared_dot8(w,aq[t+2]),shared_dot8(w,aq[t+3]));
            sums[g]+=scale*(convert_float4(vload4(0,ad+t))*dot-8.0f*convert_float4(vload4(0,asum+t)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(row<m) {
        #pragma unroll
        for(uint g=0;g<8;++g) {
            uint t=token+g*4;
            if(t<n) y[t*m+row]=sums[g].s0;
            if(t+1<n) y[(t+1)*m+row]=sums[g].s1;
            if(t+2<n) y[(t+2)*m+row]=sums[g].s2;
            if(t+3<n) y[(t+3)*m+row]=sums[g].s3;
        }
    }
}
#endif

__kernel void shared_q4_0_transpose(__global const uchar * input, ulong offset,
        __global half * output, uint k, uint n, uint padded_n) {
    uint i=get_global_id(0);
    if(i>=k*padded_n) return;
    uint t=i%padded_n,c=i/padded_n;
    output[i]=t<n?convert_half(((__global const float *)(input+offset))[t*k+c]):(half)0;
}

// Images alias the imported DMA allocation; they are not converted weights.
// Two rows per work item, four K partitions. Broadcast activation reads
// across a 64-lane wave, as in the native Adreno decode path.
#ifdef SHARED_CPU4X8
#define SHARED_K_PARTS 8
#else
#define SHARED_K_PARTS 4
#endif
SHARED_SUBGROUP
__kernel void shared_q4_0_mv(
#ifdef SHARED_CPU4X8
        __global const uint * weights,
#else
        read_only image1d_buffer_t weights,
#endif
        ulong wo,
        read_only image1d_buffer_t input, __global uchar * output, ulong yo,
        uint k, uint m) {
    uint lane=get_local_id(0), part=get_local_id(1);
    uint row=get_global_id(0)*2, safe_row=min(row,m-2), kb=k/32;
    __global float * y=(__global float *)(output+yo);
    float2 sum=0;
    for(uint b=part;b<kb;b+=SHARED_K_PARTS) {
#ifdef SHARED_CPU4X8
        uint tile=(uint)(wo/4)+((safe_row/4)*kb+b)*18;
        float2 scale=convert_float2(as_half2(weights[tile+(safe_row%4)/2]));
        uint4 qlo=vload4(0,weights+tile+2+(safe_row%4)*2)^0x88888888u;
        uint4 qhi=vload4(0,weights+tile+10+(safe_row%4)*2)^0x88888888u;
#else
        uint tile=(uint)(wo/4)+((safe_row/32)*kb+b)*144;
        uint ri=(safe_row%32)/4, si=tile+128+(safe_row%32)/2, shift=(safe_row%4)*8;
        float2 scale=convert_float2(as_half2(read_imageui(weights,si).x));
#endif
        float2 acc=0;
        float4 loaded=0;
        if(lane<8) loaded=read_imagef(input,b*8+lane);
#ifdef SHARED_CPU4X8
#define CPU4_MV(BITS,J,S) { \
        float4 a=(float4)(sub_group_broadcast(loaded.s0,J),sub_group_broadcast(loaded.s1,J), \
            sub_group_broadcast(loaded.s2,J),sub_group_broadcast(loaded.s3,J)); \
        acc+=convert_float2(convert_int2(((BITS)>>(S))&15)-8)*a.s0; \
        acc+=convert_float2(convert_int2(((BITS)>>((S)+8))&15)-8)*a.s1; \
        acc+=convert_float2(convert_int2(((BITS)>>((S)+16))&15)-8)*a.s2; \
        acc+=convert_float2(convert_int2(((BITS)>>((S)+24))&15)-8)*a.s3; \
    }
        CPU4_MV(qlo.s02,0,0)
        CPU4_MV(qlo.s13,1,0)
        CPU4_MV(qhi.s02,2,0)
        CPU4_MV(qhi.s13,3,0)
        CPU4_MV(qlo.s02,4,4)
        CPU4_MV(qlo.s13,5,4)
        CPU4_MV(qhi.s02,6,4)
        CPU4_MV(qhi.s13,7,4)
#undef CPU4_MV
#else
        // Match the native decode's constant broadcast lanes and contracted
        // multiply/add arithmetic. An explicit fma requests stricter rounding.
        #pragma unroll
        for(uint j=0;j<8;++j) {
            uint2 p=convert_uint2(as_uchar2((ushort)(read_imageui(weights,tile+j*16+ri).x>>shift)));
            uint2 q=convert_uint2(as_uchar2((ushort)(read_imageui(weights,tile+j*16+8+ri).x>>shift)));
            uint2 q0=p&15,q1=p>>4,q2=q&15,q3=q>>4;
            float4 a=(float4)(sub_group_broadcast(loaded.s0,j),sub_group_broadcast(loaded.s1,j),
                sub_group_broadcast(loaded.s2,j),sub_group_broadcast(loaded.s3,j));
            acc+=convert_float2(convert_int2(q0)-8)*a.s0;
            acc+=convert_float2(convert_int2(q1)-8)*a.s1;
            acc+=convert_float2(convert_int2(q2)-8)*a.s2;
            acc+=convert_float2(convert_int2(q3)-8)*a.s3;
        }
#endif
        sum+=scale*acc;
    }
    __local float2 partial[64*(SHARED_K_PARTS-1)];
    if(part) partial[(part-1)*64+lane]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(part==0 && row<m) {
        for(uint p=0;p<SHARED_K_PARTS-1;++p) sum+=partial[lane+p*64];
        vstore2(sum,0,y+row);
    }
}

// Same 4-row x 8-token arithmetic tile as native Adreno FP16 GEMM, with
// address generation for 32x32 shared weights. Only activations are transposed.
__kernel void shared_q4_0_mm(
#ifdef SHARED_CPU4X8
        __global const uint * weights,
#else
        read_only image1d_buffer_t weights,
#endif
        ulong wo,
        read_only image1d_buffer_t input, __global uchar * output, ulong yo,
        uint k, uint m, uint n, uint padded_n) {
    uint token=get_global_id(0)*8, row=get_global_id(1)*4;
    if(token>=n || row>=m) return;
    __global float * y=(__global float *)(output+yo);
    half8 c0=0,c1=0,c2=0,c3=0;
    for(uint b=0;b<k/32;++b) {
#ifdef SHARED_CPU4X8
        uint tile=(uint)(wo/4)+((row/4)*(k/32)+b)*18;
        half4 scale=as_half4(vload2(0,weights+tile));
        // Load the entire 64-byte quant block once. The four-row CPU
        // interleave is contiguous here, unlike the feature-major V1 image.
        uint16 packed=vload16(0,weights+tile+2)^0x88888888u;
#define CPU4_ACC(BITS, COLUMN, SHIFT) { \
        half4 w=convert_half4(convert_int4(((BITS)>>(SHIFT))&15)-8)*scale; \
        uint ai=(b*32+(COLUMN))*(padded_n/4)+token/4; \
        half8 av=(half8)(read_imageh(input,ai),read_imageh(input,ai+1)); \
        c0+=av*w.s0; c1+=av*w.s1; c2+=av*w.s2; c3+=av*w.s3; \
    }
#define CPU4_WORD(BITS, COLUMN, HIGH) \
        CPU4_ACC(BITS,COLUMN,HIGH) \
        CPU4_ACC(BITS,(COLUMN)+1,(HIGH)+8) \
        CPU4_ACC(BITS,(COLUMN)+2,(HIGH)+16) \
        CPU4_ACC(BITS,(COLUMN)+3,(HIGH)+24)
        CPU4_WORD(packed.s0246,0,0)
        CPU4_WORD(packed.s1357,4,0)
        CPU4_WORD(packed.s8ace,8,0)
        CPU4_WORD(packed.s9bdf,12,0)
        CPU4_WORD(packed.s0246,16,4)
        CPU4_WORD(packed.s1357,20,4)
        CPU4_WORD(packed.s8ace,24,4)
        CPU4_WORD(packed.s9bdf,28,4)
#undef CPU4_WORD
#undef CPU4_ACC
#else
        uint tile=(uint)(wo/4)+((row/32)*(k/32)+b)*144;
        uint si=tile+128+(row%32)/2;
        half4 scale=as_half4((uint2)(read_imageui(weights,si).x,read_imageui(weights,si+1).x));
        for(uint j=0;j<16;++j) {
            uint4 q=convert_uint4(as_uchar4(read_imageui(weights,tile+j*8+(row%32)/4).x));
            uint4 loq=q&15,hiq=q>>4;
            half4 lo=convert_half4(convert_int4(loq)-8)*scale;
            half4 hi=convert_half4(convert_int4(hiq)-8)*scale;
            uint index=(b*32+j*2)*(padded_n/4)+token/4;
            half8 a=(half8)(read_imageh(input,index),read_imageh(input,index+1));
            half8 z=(half8)(read_imageh(input,index+padded_n/4),read_imageh(input,index+padded_n/4+1));
            c0+=a*lo.s0; c1+=a*lo.s1; c2+=a*lo.s2; c3+=a*lo.s3;
            c0+=z*hi.s0; c1+=z*hi.s1; c2+=z*hi.s2; c3+=z*hi.s3;
        }
#endif
    }
    vstore4(convert_float4((half4)(c0.s0,c1.s0,c2.s0,c3.s0)),0,y+token*m+row);
    if(token+1<n) vstore4(convert_float4((half4)(c0.s1,c1.s1,c2.s1,c3.s1)),0,y+(token+1)*m+row);
    if(token+2<n) vstore4(convert_float4((half4)(c0.s2,c1.s2,c2.s2,c3.s2)),0,y+(token+2)*m+row);
    if(token+3<n) vstore4(convert_float4((half4)(c0.s3,c1.s3,c2.s3,c3.s3)),0,y+(token+3)*m+row);
    if(token+4<n) vstore4(convert_float4((half4)(c0.s4,c1.s4,c2.s4,c3.s4)),0,y+(token+4)*m+row);
    if(token+5<n) vstore4(convert_float4((half4)(c0.s5,c1.s5,c2.s5,c3.s5)),0,y+(token+5)*m+row);
    if(token+6<n) vstore4(convert_float4((half4)(c0.s6,c1.s6,c2.s6,c3.s6)),0,y+(token+6)*m+row);
    if(token+7<n) vstore4(convert_float4((half4)(c0.s7,c1.s7,c2.s7,c3.s7)),0,y+(token+7)*m+row);
}
