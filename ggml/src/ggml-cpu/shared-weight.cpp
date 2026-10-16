#include "shared-weight.h"
#include "ggml-shared-weight-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include <cstring>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static void pack_rows(const uint8_t * tiles, block_q4_0x4 * scratch, int kb) {
#if defined(__aarch64__)
    // Transpose 16 rows in registers, then store the existing 4x8 CPU format.
    // Loading complete quant planes avoids 1024 scalar nibble gathers per tile.
    for(int b=0;b<kb;++b) {
        const auto * tile=tiles+b*576;
        for(int group=0;group<8;++group) { std::memcpy(scratch[group*kb+b].d,tile+512+group*8,8); }
        for(int row=0;row<32;row+=16) {
            uint8x16_t col[16];
            for(int j=0;j<8;++j) {
                auto p=vld1q_u8(tile+j*32+row),q=vld1q_u8(tile+(j+8)*32+row);
                col[2*j]=veorq_u8(vorrq_u8(vandq_u8(p,vdupq_n_u8(15)),vshlq_n_u8(q,4)),vdupq_n_u8(0x88));
                col[2*j+1]=veorq_u8(vorrq_u8(vshrq_n_u8(p,4),vandq_u8(q,vdupq_n_u8(240))),vdupq_n_u8(0x88));
            }
            for(int h=0;h<2;++h) {
                auto * c=col+h*8;
                uint8x16x2_t a[4]={vzipq_u8(c[0],c[1]),vzipq_u8(c[2],c[3]),vzipq_u8(c[4],c[5]),vzipq_u8(c[6],c[7])};
                for(int half=0;half<2;++half) {
                    auto l=vzipq_u16(vreinterpretq_u16_u8(a[0].val[half]),vreinterpretq_u16_u8(a[1].val[half]));
                    auto r=vzipq_u16(vreinterpretq_u16_u8(a[2].val[half]),vreinterpretq_u16_u8(a[3].val[half]));
                    for(int group=0;group<2;++group) {
                        auto v=vzipq_u32(vreinterpretq_u32_u16(l.val[group]),vreinterpretq_u32_u16(r.val[group]));
                        auto * out=reinterpret_cast<uint8_t *>(scratch[(row/4+half*2+group)*kb+b].qs)+h*32;
                        vst1q_u8(out,vreinterpretq_u8_u32(v.val[0]));
                        vst1q_u8(out+16,vreinterpretq_u8_u32(v.val[1]));
                    }
                }
            }
        }
    }
#else
    for(int group=0;group<8;++group) { for(int b=0;b<kb;++b) {
        auto & out=scratch[group*kb+b]; auto * tile=tiles+b*576;
        std::memcpy(out.d,tile+512+group*8,8);
        for(int r=0;r<4;++r) { for(int j=0;j<16;++j) {
            unsigned shift=(j%2)*4;
            uint8_t v=((tile[(j/2)*32+group*4+r]>>shift)&15) |
                      (((tile[(j/2+8)*32+group*4+r]>>shift)&15)<<4);
            out.qs[(j/8)*32+r*8+j%8]=int8_t(v^0x88);
        } }
    } }
#endif
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static void direct_mv(const uint8_t * tiles, const block_q8_0 * x, float * dst, int kb) {
    float32x4_t sums[8]; for(auto & s:sums) { s=vdupq_n_f32(0); }
    for(int b=0;b<kb;++b) {
        auto * tile=tiles+b*576;
        int32x4_t dots[8]; for(auto & d:dots) { d=vdupq_n_s32(0); }
        for(int j=0;j<8;++j) {
            int32_t abits; std::memcpy(&abits,x[b].qs+j*4,4);
            int8x16_t a=vreinterpretq_s8_s32(vdupq_n_s32(abits));
            for(int h=0;h<2;++h) {
                auto p=vld1q_u8(tile+(j*2)*32+h*16),q=vld1q_u8(tile+(j*2+1)*32+h*16);
                auto p0=vsubq_u8(vandq_u8(p,vdupq_n_u8(15)),vdupq_n_u8(8));
                auto p1=vsubq_u8(vshrq_n_u8(p,4),vdupq_n_u8(8));
                auto q0=vsubq_u8(vandq_u8(q,vdupq_n_u8(15)),vdupq_n_u8(8));
                auto q1=vsubq_u8(vshrq_n_u8(q,4),vdupq_n_u8(8));
                auto pl=vzipq_u8(p0,p1),ql=vzipq_u8(q0,q1);
                for(int half=0;half<2;++half) {
                    auto v=vzipq_u16(vreinterpretq_u16_u8(pl.val[half]),vreinterpretq_u16_u8(ql.val[half]));
                    dots[h*4+half*2]=vdotq_s32(dots[h*4+half*2],vreinterpretq_s8_u16(v.val[0]),a);
                    dots[h*4+half*2+1]=vdotq_s32(dots[h*4+half*2+1],vreinterpretq_s8_u16(v.val[1]),a);
                }
            }
        }
        float d=GGML_FP16_TO_FP32(x[b].d);
        for(int group=0;group<8;++group) {
            auto scales=vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(reinterpret_cast<const uint16_t *>(tile+512+group*8))));
            sums[group]=vfmaq_f32(sums[group],vcvtq_f32_s32(dots[group]),vmulq_n_f32(scales,d));
        }
    }
    for(int group=0;group<8;++group) { vst1q_f32(dst+group*4,sums[group]); }
}
#endif

// Keep at most 32 output rows in thread-local packed scratch. No allocation or
// weight-sized cache occurs in compute. Conversion remains inside the operator.
class shared_traits final : public ggml::cpu::tensor_traits {
public:
    bool work_size(int nth, const ggml_tensor * op, size_t & size) override {
        size_t k=op->src[0]->ne[0], n=op->src[1]->ne[1];
        size_t qrow=ggml_row_size(GGML_TYPE_Q8_0,k), scratch=GGML_PAD(k*18,128);
        if(nth<=0 || !n || n>INT32_MAX || n>(SIZE_MAX-127)/qrow ||
           size_t(nth)>(SIZE_MAX-GGML_PAD(qrow*n,128))/scratch) { return false; }
        size=GGML_PAD(qrow*n,128)+size_t(nth)*scratch;
        return true;
    }
    bool compute_forward(ggml_compute_params * p, ggml_tensor * op) override {
        const auto * w=op->src[0]; const auto * x=op->src[1];
        int k=w->ne[0], m=w->ne[1], n=x->ne[1], kb=k/32;
        size_t qrow=ggml_row_size(GGML_TYPE_Q8_0,k), qbytes=GGML_PAD(qrow*n,128);
        auto * act=static_cast<uint8_t *>(p->wdata);
        auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
        int grouped=n/4*4;
        for(int t=p->ith*4;t<grouped;t+=p->nth*4) {
            ggml_quantize_mat_q8_0_4x8(static_cast<const float *>(x->data)+size_t(t)*k,act+size_t(t)*qrow,k);
        }
        for(int t=grouped+p->ith;t<n;t+=p->nth) {
            quant(static_cast<const float *>(x->data)+size_t(t)*k,act+size_t(t)*qrow,k);
        }
        ggml_barrier(p->threadpool);
        auto * scratch=reinterpret_cast<block_q4_0x4 *>(act+qbytes+size_t(p->ith)*GGML_PAD(size_t(k)*18,128));
        for(int ct=p->ith;ct<m/32;ct+=p->nth) {
            auto * tiles=static_cast<const uint8_t *>(w->data)+size_t(ct)*kb*576;
            auto * dst=static_cast<float *>(op->data)+ct*32;
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
            if(n==1) { direct_mv(tiles,reinterpret_cast<const block_q8_0 *>(act),dst,kb); continue; }
#endif
            pack_rows(tiles,scratch,kb);
            if(grouped) { ggml_gemm_q4_0_4x8_q8_0(k,dst,m,scratch,act,grouped,32); }
            for(int t=grouped;t<n;++t) {
                ggml_gemv_q4_0_4x8_q8_0(k,dst+size_t(t)*m,m,scratch,act+size_t(t)*qrow,1,32);
            }
        }
        return true;
    }
};

static bool supported(const ggml_tensor * op) {
    auto * w=op->src[0]; auto * x=op->src[1];
    return op->op==GGML_OP_MUL_MAT && w && x && w->buffer &&
        w->buffer->buft==ggml_backend_cpu_shared_weight_buffer_type() &&
        w->type==GGML_TYPE_Q4_0 && w->ne[0]%32==0 && w->ne[1]%32==0 &&
        w->ne[2]==1 && w->ne[3]==1 && !w->view_src && ggml_is_contiguous(w) &&
        op->type==GGML_TYPE_F32 && ggml_is_contiguous(op) &&
        x->type==GGML_TYPE_F32 && x->ne[1]>0 && x->ne[1]<=INT32_MAX &&
        x->ne[0]==w->ne[0] && x->ne[2]==1 && x->ne[3]==1 &&
        ggml_is_contiguous(x) && (!x->buffer || ggml_backend_buft_is_host(x->buffer->buft));
}
class shared_extra final : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override { return supported(op); }
    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        static shared_traits traits;
        return supported(op)?&traits:nullptr;
    }
};
ggml_backend_buffer_type_t ggml_backend_cpu_shared_weight_buffer_type() {
    static shared_extra extra;
    static ggml_backend_buffer_type type={
        {[](ggml_backend_buffer_type_t) { return "CPU_SHARED_Q4_0_32X32"; },
         [](ggml_backend_buffer_type_t,size_t)->ggml_backend_buffer_t { return nullptr; },
         [](ggml_backend_buffer_type_t)->size_t { return 128; },nullptr,nullptr,nullptr},
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(),0),&extra};
    return &type;
}
ggml_backend_buffer_t ggml_backend_cpu_import_shared_weight(ggml_backend_t backend, const ggml_shared_weight_desc * desc) {
    if(!backend || !ggml_backend_is_cpu(backend) || !ggml_shared_weight_desc_valid(desc)) { return nullptr; }
    // Static initialization is thread-safe; construction below does not mutate it.
    static const ggml_backend_buffer_i immutable_iface=[] {
        ggml_backend_buffer_i i={};
        i.free_buffer=[](ggml_backend_buffer_t b) { delete static_cast<ggml_shared_weight_desc *>(b->context); };
        i.get_base=[](ggml_backend_buffer_t b) { return static_cast<ggml_shared_weight_desc *>(b->context)->storage.host_ptr; };
        i.init_tensor=[](ggml_backend_buffer_t b,ggml_tensor * t) {
            auto * d=static_cast<ggml_shared_weight_desc *>(b->context);
            return !t->view_src && t->type==GGML_TYPE_Q4_0 && t->ne[0]==d->k && t->ne[1]==d->m &&
                t->ne[2]==1 && t->ne[3]==1 && ggml_is_contiguous(t) &&
                t->data==static_cast<char *>(d->storage.host_ptr)+d->offset?GGML_STATUS_SUCCESS:GGML_STATUS_FAILED;
        };
        i.get_tensor=[](ggml_backend_buffer_t,const ggml_tensor * t,void * out,size_t off,size_t size) {
            ggml_shared_weight_get_canonical(t->data,t->ne[0],out,off,size);
        };
        return i;
    }();
    return ggml_backend_buffer_init(ggml_backend_cpu_shared_weight_buffer_type(),immutable_iface,
                                    new ggml_shared_weight_desc(*desc),desc->storage.size);
}
