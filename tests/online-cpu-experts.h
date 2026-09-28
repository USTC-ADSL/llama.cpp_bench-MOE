#pragma once

// Uses one bounded target, shared by every layer. The immutable canonical source
// stays CPU/NPU-readable. Caller serializes CPU work and owns activation storage.
struct online_cpu_experts {
    device & cpu;
    std::string mode;
    int capacity;
    bank target;
    std::vector<uint8_t> staging;
    size_t target_bytes;

    online_cpu_experts(device &d,const bank &source,const std::string &kind)
        :cpu(d),mode(kind),capacity(kind=="repack-id"?8:1),target(d,source,capacity,kind!="canonical-mm") {
        require(mode=="repack-id" || mode=="repack-mm" || mode=="canonical-mm","invalid CPU execution mode");
        target_bytes=target.buf?ggml_backend_buffer_get_size(target.buf.get()):0;
        if(capacity>1) {
            size_t bytes=0; for(auto *w:target.w) { bytes=std::max(bytes,ggml_nbytes(w)); }
            staging.resize(bytes);
        }
    }
    size_t bytes() const { return target_bytes+staging.capacity(); }

    double prepare(const bank &source,const std::vector<int> &experts) {
        require(!experts.empty() && experts.size()<=size_t(capacity),"CPU repack group capacity");
        const auto start=timer::now();
        for(int i=0;i<3;++i) {
            auto *s=source.w[i],*t=target.w[i];
            require(s->type==t->type && s->ne[0]==t->ne[0] && s->ne[1]==t->ne[1] && s->data &&
                    ggml_is_contiguous(s) && s->nb[2]==t->nb[2],"CPU repack source changed");
            for(int e:experts) { require(e>=0 && e<s->ne[2],"CPU repack expert bounds"); }
            const auto *base=static_cast<const uint8_t *>(s->data);
            const void *raw=base+experts[0]*s->nb[2];
            if(mode=="canonical-mm") {
                require(ggml_backend_buffer_is_host(s->buffer),"canonical CPU view needs host backing");
                // Borrow a complete contiguous matrix. Source banks outlive the
                // executor and each graph has completed before this is rebound.
                t->buffer=s->buffer; t->data=const_cast<void *>(raw); t->extra=nullptr;
                continue;
            }
            if(capacity>1) {
                // Valid repeated padding; no route can reference unused slots.
                for(int slot=0;slot<capacity;++slot) {
                    int e=experts[std::min(size_t(slot),experts.size()-1)];
                    std::memcpy(staging.data()+slot*s->nb[2],base+e*s->nb[2],s->nb[2]);
                }
                raw=staging.data();
            }
            ggml_backend_tensor_set(t,raw,0,ggml_nbytes(t));
        }
        cpu.sync(); return elapsed(start);
    }

    json run(const bank &source,const float *x,int top_k,const std::vector<int32_t> &ids,
             const std::vector<int> &pairs,int nt,const std::shared_ptr<shared_activation_buffer> &arena,
             const std::function<void(int,const float *)> &consume) {
        const auto start=timer::now();
        const int width=int(source.w[0]->ne[0]),out_width=int(source.w[2]->ne[1]);
        std::vector<std::vector<int>> by_expert(source.w[0]->ne[2]);
        for(int p:pairs) {
            require(p>=0 && p<int(ids.size()) && ids[p]>=0 && ids[p]<int(by_expert.size()),"CPU group route bounds");
            by_expert[ids[p]].push_back(p);
        }
        std::vector<int> active;
        for(size_t e=0;e<by_expert.size();++e) { if(!by_expert[e].empty()) { active.push_back(int(e)); } }
        double group_us=elapsed(start),conversion=0,setup=0,compute=0,scatter=0;
        int batches=0,prepared=0;
        for(size_t begin=0;begin<active.size();begin+=capacity) {
            std::vector<int> experts(active.begin()+begin,active.begin()+std::min(active.size(),begin+capacity));
            conversion+=prepare(source,experts); ++batches; prepared+=capacity;
            auto setup_start=timer::now(); std::vector<int> group,local;
            for(size_t slot=0;slot<experts.size();++slot) {
                for(int p:by_expert[experts[slot]]) { group.push_back(p); local.push_back(int(slot)); }
            }
            oracle_job job(target,nullptr,int(group.size()),1,nt,false,mode!="repack-id",true,arena,false,true);
            for(size_t row=0;row<group.size();++row) {
                std::copy_n(x+(group[row]/top_k)*width,width,job.input_data()+row*width);
                job.index_data()[row]=local[row];
            }
            setup+=elapsed(setup_start); compute+=job.run()[1]; auto scatter_start=timer::now();
            for(size_t row=0;row<group.size();++row) { consume(group[row],job.output_data()+row*out_width); }
            scatter+=elapsed(scatter_start);
        }
        return {{"mode",mode},{"group_us",group_us},{"conversion_us",conversion},{"setup_gather_us",setup},
            {"prepare_us",group_us+conversion+setup},{"compute_us",compute},{"scatter_us",scatter},
            {"total_us",elapsed(start)},{"active_experts",active.size()},{"groups",batches},
            {"prepared_slots",prepared},{"target_bytes",target_bytes},{"staging_bytes",staging.capacity()}};
    }
};
