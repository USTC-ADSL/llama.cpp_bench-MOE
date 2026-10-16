#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define SHARED_SUBGROUP __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define SHARED_SUBGROUP
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
SHARED_SUBGROUP
__kernel void shared_q4_0_mv(read_only image1d_buffer_t weights, ulong wo,
        read_only image1d_buffer_t input, __global uchar * output, ulong yo,
        uint k, uint m) {
    uint lane=get_local_id(0), part=get_local_id(1);
    uint row=get_global_id(0)*2, safe_row=min(row,m-2), kb=k/32;
    __global float * y=(__global float *)(output+yo);
    float2 sum=0;
    for(uint b=part;b<kb;b+=4) {
        uint tile=(uint)(wo/4)+((safe_row/32)*kb+b)*144;
        uint ri=(safe_row%32)/4, si=tile+128+(safe_row%32)/2, shift=(safe_row%4)*8;
        float2 scale=convert_float2(as_half2(read_imageui(weights,si).x));
        float2 acc=0;
        float4 loaded=0;
        if(lane<8) loaded=read_imagef(input,b*8+lane);
        // Match the native decode's constant broadcast lanes and contracted
        // multiply/add arithmetic. An explicit fma requests stricter rounding.
        #pragma unroll
        for(uint j=0;j<8;++j) {
            uint2 p=convert_uint2(as_uchar2((ushort)(read_imageui(weights,tile+j*16+ri).x>>shift)));
            uint2 q=convert_uint2(as_uchar2((ushort)(read_imageui(weights,tile+j*16+8+ri).x>>shift)));
            float4 a=(float4)(sub_group_broadcast(loaded.s0,j),sub_group_broadcast(loaded.s1,j),
                sub_group_broadcast(loaded.s2,j),sub_group_broadcast(loaded.s3,j));
            acc+=convert_float2(convert_int2(p&15)-8)*a.s0;
            acc+=convert_float2(convert_int2(p>>4)-8)*a.s1;
            acc+=convert_float2(convert_int2(q&15)-8)*a.s2;
            acc+=convert_float2(convert_int2(q>>4)-8)*a.s3;
        }
        sum+=scale*acc;
    }
    __local float2 partial[192];
    if(part) partial[(part-1)*64+lane]=sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(part==0 && row<m) {
        for(uint p=0;p<3;++p) sum+=partial[lane+p*64];
        vstore2(sum,0,y+row);
    }
}

// Same 4-row x 8-token arithmetic tile as native Adreno FP16 GEMM, with
// address generation for 32x32 shared weights. Only activations are transposed.
__kernel void shared_q4_0_mm(read_only image1d_buffer_t weights, ulong wo,
        read_only image1d_buffer_t input, __global uchar * output, ulong yo,
        uint k, uint m, uint n, uint padded_n) {
    uint token=get_global_id(0)*8, row=get_global_id(1)*4;
    if(token>=n || row>=m) return;
    __global float * y=(__global float *)(output+yo);
    half8 c0=0,c1=0,c2=0,c3=0;
    for(uint b=0;b<k/32;++b) {
        uint tile=(uint)(wo/4)+((row/32)*(k/32)+b)*144;
        uint si=tile+128+(row%32)/2;
        half4 scale=as_half4((uint2)(read_imageui(weights,si).x,read_imageui(weights,si+1).x));
        for(uint j=0;j<16;++j) {
            uint4 q=convert_uint4(as_uchar4(read_imageui(weights,tile+j*8+(row%32)/4).x));
            half4 lo=convert_half4(convert_int4(q&15)-8)*scale;
            half4 hi=convert_half4(convert_int4(q>>4)-8)*scale;
            uint index=(b*32+j*2)*(padded_n/4)+token/4;
            half8 a=(half8)(read_imageh(input,index),read_imageh(input,index+1));
            half8 z=(half8)(read_imageh(input,index+padded_n/4),read_imageh(input,index+padded_n/4+1));
            c0+=a*lo.s0; c1+=a*lo.s1; c2+=a*lo.s2; c3+=a*lo.s3;
            c0+=z*hi.s0; c1+=z*hi.s1; c2+=z*hi.s2; c3+=z*hi.s3;
        }
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
