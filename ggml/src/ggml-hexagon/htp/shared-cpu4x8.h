#pragma once

// Only VTCM staging is read with 128-byte unaligned loads. Its existing
// 640-byte-per-K-tile budget exceeds the 576-byte raw representation,
// including the last block's 56-byte load overhang (K >= 128).
typedef unsigned char cpu4x8_bytes128 __attribute__((vector_size(128)));

static inline HVX_Vector cpu4x8_unpack_transposed(HVX_Vector bytes) {
    bytes=Q6_V_vxor_VV(bytes,Q6_Vb_vsplat_R(0x88));
    HVX_VectorPred low=Q6_Q_vsetq_R(64);
    HVX_Vector q=Q6_V_vmux_QVV(low,Q6_V_vand_VV(bytes,Q6_Vb_vsplat_R(15)),Q6_Vub_vlsr_VubR(bytes,4));
    return Q6_Vb_vsub_VbVb(q,Q6_Vb_vsplat_R(8));
}

static inline HVX_Vector cpu4x8_decode_dot(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    return cpu4x8_unpack_transposed((HVX_Vector)__builtin_shufflevector(raw,raw,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,12,13,14,15,20,21,22,23,28,29,30,31,36,37,38,39,40,41,42,43,48,49,50,51,56,57,58,59,64,65,66,67,44,45,46,47,52,53,54,55,60,61,62,63,68,69,70,71,8,9,10,11,16,17,18,19,24,25,26,27,32,33,34,35,12,13,14,15,20,21,22,23,28,29,30,31,36,37,38,39,40,41,42,43,48,49,50,51,56,57,58,59,64,65,66,67,44,45,46,47,52,53,54,55,60,61,62,63,68,69,70,71));
}

static inline HVX_Vector cpu4x8_decode_hmx(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    return cpu4x8_unpack_transposed((HVX_Vector)__builtin_shufflevector(raw,raw,8,9,16,17,24,25,32,33,10,11,18,19,26,27,34,35,12,13,20,21,28,29,36,37,14,15,22,23,30,31,38,39,40,41,48,49,56,57,64,65,42,43,50,51,58,59,66,67,44,45,52,53,60,61,68,69,46,47,54,55,62,63,70,71,8,9,16,17,24,25,32,33,10,11,18,19,26,27,34,35,12,13,20,21,28,29,36,37,14,15,22,23,30,31,38,39,40,41,48,49,56,57,64,65,42,43,50,51,58,59,66,67,44,45,52,53,60,61,68,69,46,47,54,55,62,63,70,71));
}

static inline HVX_Vector cpu4x8_hmx_scales(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    return (HVX_Vector)__builtin_shufflevector(raw,raw,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7);
}

static inline HVX_Vector cpu4x8_decode_i8(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    HVX_Vector bytes=(HVX_Vector)__builtin_shufflevector(raw,raw,8,9,10,11,12,13,14,15,40,41,42,43,44,45,46,47,8,9,10,11,12,13,14,15,40,41,42,43,44,45,46,47,16,17,18,19,20,21,22,23,48,49,50,51,52,53,54,55,16,17,18,19,20,21,22,23,48,49,50,51,52,53,54,55,24,25,26,27,28,29,30,31,56,57,58,59,60,61,62,63,24,25,26,27,28,29,30,31,56,57,58,59,60,61,62,63,32,33,34,35,36,37,38,39,64,65,66,67,68,69,70,71,32,33,34,35,36,37,38,39,64,65,66,67,68,69,70,71);
    bytes=Q6_V_vxor_VV(bytes,Q6_Vb_vsplat_R(0x88));
    static const uint8_t high_bytes[128] __attribute__((aligned(128)))={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
    HVX_VectorPred high=Q6_Q_vcmp_gt_VubVub(hvx_vmem(high_bytes),Q6_V_vzero());
    HVX_Vector q=Q6_V_vmux_QVV(high,Q6_Vub_vlsr_VubR(bytes,4),Q6_V_vand_VV(bytes,Q6_Vb_vsplat_R(15)));
    return Q6_Vb_vsub_VbVb(q,Q6_Vb_vsplat_R(8));
}

static inline HVX_Vector cpu4x8_dot_scales(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    return (HVX_Vector)__builtin_shufflevector(raw,raw,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7);
}

static inline HVX_Vector cpu4x8_half_scales(HVX_Vector v) {
    const cpu4x8_bytes128 raw=(cpu4x8_bytes128)v;
    return (HVX_Vector)__builtin_shufflevector(raw,raw,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3);
}
