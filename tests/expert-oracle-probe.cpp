// Reuse the established model loader and complete attention/router prefix.
#define main legacy_routing_main
#include "expert-routing-probe.cpp"
#undef main
#include "shared-activation-buffer.h"

struct oracle_job {
    device & dev;
    ggml_context_ptr ctx = context(64, 64);
    ggml_backend_buffer_ptr buf;
    std::shared_ptr<shared_activation_buffer> shared_buffer;
    bool pooled_activations=false;
    ggml_tensor * x = nullptr, * ids_storage = nullptr, * out = nullptr;
    ggml_tensor * sx = nullptr, * sout = nullptr, * packed_map = nullptr, * input_rows = nullptr;
    ggml_cgraph * graph;
    std::vector<float> input, output, shared_input, shared_output;
    std::vector<int32_t> indices;
    int stride = 0;
    oracle_job(bank & b, shared_bank * sw, int rows, int k, int nt, bool native, bool single_matmul = false, bool shared_activations = false,
               std::shared_ptr<shared_activation_buffer> arena = {}, bool layout_only = false, bool reuse_lifetimes = false,
               const shared_activation_layout::reservation * reserved = nullptr,
               bank * packed = nullptr, const std::vector<int> * packed_ids = nullptr, bool device_gather = false) : dev(b.d) {
        require(!device_gather || (dev.name=="HTP0" && shared_activations && rows>0 && k==1 && !native),
                "device gather requires split HTP shared activations");
        graph = ggml_new_graph_custom(ctx.get(), 64, false);
        if (rows) {
            x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, b.w[0]->ne[0], 1, device_gather?nt:rows);
            auto * routed_input=x;
            if(device_gather) {
                input_rows=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,rows);
                auto * gathered=ggml_get_rows(ctx.get(),ggml_reshape_2d(ctx.get(),x,x->ne[0],nt),input_rows);
                routed_input=ggml_reshape_3d(ctx.get(),gathered,x->ne[0],1,rows);
            }
            stride = dev.name == "GPUOpenCL" ? int(b.w[0]->ne[2]) : k;
            ids_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, stride, rows);
            auto * ids = ggml_view_2d(ctx.get(), ids_storage, k, rows, ids_storage->nb[1], 0);
            if (single_matmul && b.w[0]->ne[2] == 1) {
                require(k == 1, "single Expert requires one route per gathered row");
                auto * matrix = ggml_reshape_2d(ctx.get(), routed_input, x->ne[0], rows);
                auto * u = ggml_mul_mat(ctx.get(), b.w[1], matrix);
                auto * g = ggml_mul_mat(ctx.get(), b.w[0], matrix);
                out = ggml_mul_mat(ctx.get(), b.w[2], ggml_swiglu_split(ctx.get(), g, u));
            } else {
                auto * u = ggml_mul_mat_id(ctx.get(), b.w[1], routed_input, ids);
                auto * g = ggml_mul_mat_id(ctx.get(), b.w[0], routed_input, ids);
                out = ggml_mul_mat_id(ctx.get(), b.w[2], ggml_swiglu_split(ctx.get(), g, u), ids);
                if (packed) {
                    require(dev.name == "HTP0" && b.canonical_weights && packed_ids && shared_activations,
                            "mixed Expert graph needs canonical HTP weights and shared activations");
                    packed_map = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, b.w[0]->ne[2]);
                    ggml_set_name(packed_map, "packed_expert_map");
                    ggml_tensor * ops[] = {g, u, out};
                    for (int i = 0; i < 3; ++i) { ops[i]->src[3] = packed->w[i]; ops[i]->src[4] = packed_map; }
                }
            }
            ggml_build_forward_expand(graph, out);
            if (!shared_activations) {
                input.resize(ggml_nelements(x)); indices.resize(stride*rows);
                output.resize(ggml_nelements(out));
            }
        }
        if (sw) {
            sx = (native || device_gather) && x ? ggml_reshape_2d(ctx.get(), x, x->ne[0], nt) :
                ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, b.w[0]->ne[0], nt);
            auto * u = ggml_mul_mat(ctx.get(), sw->w[1], sx);
            auto * g = ggml_mul_mat(ctx.get(), sw->w[0], sx);
            auto * d = ggml_mul_mat(ctx.get(), sw->w[2], ggml_swiglu_split(ctx.get(), g, u));
            auto * r = ggml_mul_mat(ctx.get(), sw->w[3], sx);
            sout = ggml_mul(ctx.get(), d, ggml_div(ctx.get(), ggml_silu(ctx.get(), r), r));
            ggml_build_forward_expand(graph, sout);
            if (!shared_activations) { shared_input.resize(ggml_nelements(sx)); shared_output.resize(ggml_nelements(sout)); }
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(dev.backend.get(), node)) {
                fprintf(stderr, "%s unsupported %s\n", dev.name.c_str(), ggml_op_name(node->op));
                throw std::runtime_error("unsupported oracle graph");
            }
        }
        require((!arena && !layout_only) || shared_activations,"arena/layout requires shared activations");
        if(layout_only) { return; }
        if (shared_activations) {
            pooled_activations=bool(arena);
            if(arena) { shared_buffer=std::move(arena); shared_buffer->bind(dev,ctx.get(),reuse_lifetimes?graph:nullptr,reserved); }
            else { shared_buffer=shared_activation_buffer::allocate(dev,ctx.get()); }
            if (ids_storage) { std::memset(ids_storage->data,0,ggml_nbytes(ids_storage)); }
            if (packed_map) {
                auto * map = static_cast<int32_t *>(packed_map->data);
                std::fill_n(map, packed_map->ne[0], -1);
                require(packed_ids->size() <= size_t(packed->w[0]->ne[2]), "packed map capacity");
                for (size_t i = 0; i < packed_ids->size(); ++i) {
                    int e = packed_ids->at(i); require(e >= 0 && e < packed_map->ne[0] && map[e] == -1, "invalid packed expert map");
                    map[e] = int32_t(i);
                }
            }
        } else {
            buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(dev.dev)));
            require(buf != nullptr, "oracle activation allocation failed");
        }
    }
    void reset_index_padding() {
        if(pooled_activations && ids_storage) { std::memset(ids_storage->data,0,ggml_nbytes(ids_storage)); }
    }
    float * input_data() { return shared_buffer?static_cast<float *>(x->data):input.data(); }
    int32_t * index_data() { return shared_buffer?static_cast<int32_t *>(ids_storage->data):indices.data(); }
    const float * output_data() const { return shared_buffer?static_cast<float *>(out->data):output.data(); }
    const float * shared_output_data() const { return shared_buffer?static_cast<float *>(sout->data):shared_output.data(); }
    void set_shared_input(const std::vector<float> & values) {
        set_shared_input(values.data(),values.size());
    }
    void set_shared_input(const float * values,size_t count) {
        if (sx && !sx->view_src) {
            require(count==size_t(ggml_nelements(sx)),"shared input size mismatch");
            std::copy_n(values,count,shared_buffer?static_cast<float *>(sx->data):shared_input.data());
        }
    }
    std::array<double, 3> run() {
        require(buf || shared_buffer,"oracle activation buffers are unbound");
        auto start = timer::now();
        if (shared_buffer) {
            dev.compute(graph);
            return {0,elapsed(start),0};
        }
        if (x) {
            ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
            ggml_backend_tensor_set(ids_storage, indices.data(), 0, indices.size()*sizeof(int32_t));
        }
        if (sx && !sx->view_src) { ggml_backend_tensor_set(sx, shared_input.data(), 0, shared_input.size()*sizeof(float)); }
        dev.sync(); double upload = elapsed(start); start = timer::now();
        dev.compute(graph); double compute = elapsed(start); start = timer::now();
        if (out) { ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float)); }
        if (sout) { ggml_backend_tensor_get(sout, shared_output.data(), 0, shared_output.size()*sizeof(float)); }
        dev.sync(); return {upload, compute, elapsed(start)};
    }
};

struct placement {
    std::string name;
    std::vector<int> owner;
    int shared;
};

struct oracle_layer {
    layer base;
    bank gpu;
    shared_bank cs, ns, gs;
    std::vector<int32_t> ids, counts;
    std::vector<int> pairs[3];
    std::unique_ptr<oracle_job> work[3];
    bool native = false;
    placement current;
    oracle_layer(device & c, device & n, device & g, source & s, int il, int k, float eps, int nt, const json * ref) :
        base(c, n, s, il, k, eps, nt, false, ref), gpu(g, s, il), cs(c, s, il), ns(n, s, il), gs(g, s, il) {}
    void prepare(const placement & p) {
        current = p;
        for (int b = 0; b < 3; ++b) { work[b].reset(); pairs[b].clear(); }
        for (int i = 0; i < int(ids.size()); ++i) { pairs[p.owner[ids[i]]].push_back(i); }
        native = false;
        for (int b = 0; b < 3; ++b) { native |= pairs[b].size() == ids.size(); }
        bank * banks[] = {&base.cb, &base.nb, &gpu};
        shared_bank * shared[] = {&cs, &ns, &gs};
        for (int b = 0; b < 3; ++b) {
            if (pairs[b].empty() && p.shared != b) { continue; }
            int rows = pairs[b].empty() ? 0 : native ? base.nt : int(pairs[b].size());
            work[b] = std::make_unique<oracle_job>(*banks[b], p.shared == b ? shared[b] : nullptr,
                                                  rows, native ? base.topk : 1, base.nt, native);
        }
    }
    std::vector<float> run(const std::vector<float> & input, json * report = nullptr) {
        auto start = timer::now();
        ggml_backend_tensor_set(base.x, input.data(), 0, input.size()*sizeof(float));
        base.cpu.compute(&base.prefix_graph);
        ggml_backend_tensor_get(base.normalized, base.activation.data(), 0, base.activation.size()*sizeof(float));
        ggml_backend_tensor_get(base.probs, base.probabilities.data(), 0, base.probabilities.size()*sizeof(float));
        ggml_backend_tensor_get(base.residual, base.residual_values.data(), 0, base.residual_values.size()*sizeof(float));
        base.cpu.sync();
        // Real top-k is still evaluated; fixed IDs define the offline replay workload.
        std::vector<int> order(base.experts);
        int changed = 0;
        for (int t = 0; t < base.nt; ++t) {
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin()+base.topk, order.end(), [&](int a, int b) {
                float pa = base.probabilities[t*base.experts+a], pb = base.probabilities[t*base.experts+b];
                return pa == pb ? a < b : pa > pb;
            });
            for (int k = 0; k < base.topk; ++k) { changed += order[k] != ids[t*base.topk+k]; }
        }
        double prefix = elapsed(start); start = timer::now();
        for (int b = 0; b < 3; ++b) {
            if (!work[b]) { continue; }
            auto & j = *work[b];
            j.shared_input = base.activation;
            if (native && j.x) {
                j.input = base.activation;
                for (int t = 0; t < base.nt; ++t) {
                    for (int k = 0; k < base.topk; ++k) { j.indices[t*j.stride+k] = ids[t*base.topk+k]; }
                }
            } else {
                for (size_t i = 0; i < pairs[b].size(); ++i) {
                    int p = pairs[b][i];
                    std::copy_n(base.activation.data()+(p/base.topk)*base.width, base.width, j.input.data()+i*base.width);
                    j.indices[i*j.stride] = ids[p];
                }
            }
        }
        double partition = elapsed(start); start = timer::now();
        std::future<std::array<double,3>> tasks[3];
        std::array<double,3> phases[3]{};
        std::array<double,2> intervals[3]{};
        int inline_backend = -1;
        for (int b = 0; b < 3; ++b) { if (work[b]) { inline_backend = b; } }
        for (int b = 0; b < 3; ++b) {
            if (work[b] && b != inline_backend) {
                tasks[b] = std::async(std::launch::async, [&, b] {
                    intervals[b][0] = elapsed(start);
                    auto result = work[b]->run();
                    intervals[b][1] = elapsed(start); return result;
                });
            }
        }
        intervals[inline_backend][0] = elapsed(start);
        phases[inline_backend] = work[inline_backend]->run();
        intervals[inline_backend][1] = elapsed(start);
        for (int b = 0; b < 3; ++b) { if (tasks[b].valid()) { phases[b] = tasks[b].get(); } }
        double experts = elapsed(start); start = timer::now();
        std::vector<const float *> outputs(ids.size());
        for (int b = 0; b < 3; ++b) {
            for (size_t i = 0; i < pairs[b].size(); ++i) { outputs[pairs[b][i]] = work[b]->output.data()+i*base.width; }
        }
        std::vector<float> result(input.size(), 0);
        for (size_t p = 0; p < ids.size(); ++p) {
            float weight = base.probabilities[(p/base.topk)*base.experts+ids[p]];
            for (int c = 0; c < base.width; ++c) { result[(p/base.topk)*base.width+c] += weight*outputs[p][c]; }
        }
        for (size_t i = 0; i < result.size(); ++i) { result[i] = base.residual_values[i]+(result[i]+work[current.shared]->shared_output[i]); }
        double combine = elapsed(start);
        if (report) {
            *report = {{"attention_router_us", prefix}, {"partition_us", partition}, {"experts_wall_us", experts},
                {"combine_us", combine}, {"phases_us", phases}, {"backend_intervals_us", intervals}, {"changed_topk_slots", changed},
                {"pairs", {pairs[0].size(), pairs[1].size(), pairs[2].size()}}};
        }
        return result;
    }
};

#ifndef EXPERT_ORACLE_NO_MAIN
int main(int argc, char ** argv) {
    try {
        require(argc == 5 || argc == 6, "usage: expert-oracle-probe MODEL REFERENCE_JSON OUTPUT_JSON REPS [FROZEN_SEARCH_JSON]");
        std::ifstream rf(argv[2]); json ref; rf >> ref;
        json frozen;
        if (argc == 6) { std::ifstream f(argv[5]); f >> frozen; }
        auto tokens = ref.at("tokens").get<std::vector<int32_t>>();
        int nt = tokens.size(), reps = std::stoi(argv[4]); require(reps >= 4, "at least four confirmation rounds required");
        ggml_context * raw = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &raw})); ggml_context_ptr meta(raw);
        require(gguf && meta, "GGUF open failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose); require(file != nullptr, "model open failed");
        source src{gguf.get(), meta.get(), file.get()};
        int topk = gguf_get_val_u32(gguf.get(), gguf_find_key(gguf.get(), "qwen2moe.expert_used_count"));
        float eps = gguf_get_val_f32(gguf.get(), gguf_find_key(gguf.get(), "qwen2moe.attention.layer_norm_rms_epsilon"));
        ggml_backend_load_all(); device cpu("CPU"), npu("HTP0"), gpu("GPUOpenCL");
        std::vector<std::unique_ptr<oracle_layer>> layers;
        for (int il = 0; il < 2; ++il) { layers.push_back(std::make_unique<oracle_layer>(cpu,npu,gpu,src,il,topk,eps,nt,ref.contains("prefix_length") ? &ref : nullptr)); }
        int width = layers[0]->base.width, ne = layers[0]->base.experts;
        auto * emb = src.tensor("token_embd.weight");
        auto offset = gguf_get_data_offset(gguf.get())+gguf_get_tensor_offset(gguf.get(),gguf_find_tensor(gguf.get(),"token_embd.weight"));
        std::vector<uint8_t> row(emb->nb[1]); std::vector<float> input(nt*width);
        for (int t = 0; t < nt; ++t) {
            require(fseeko(file.get(), off_t(offset+tokens[t]*row.size()), SEEK_SET) == 0, "embedding seek failed");
            require(fread(row.data(),1,row.size(),file.get()) == row.size(), "embedding read failed");
            ggml_get_type_traits(emb->type)->to_float(row.data(),input.data()+t*width,width);
        }
        json report = {{"tokens", nt}, {"prefix_length", ref.value("prefix_length",0)}, {"backend_order", {"CPU","HTP0","GPUOpenCL"}},
            {"scope", "offline fixed CPU routes; CPU attention/router; host aggregate; resident weights; complete dependent 1/2 layers; search excludes initialization; no LM head"},
            {"search", json::array()}, {"confirmation", json::array()}};
        auto save = [&] { std::ofstream out(argv[3]); out << report.dump(2) << '\n'; require(out.good(),"report write failed"); };
        std::vector<placement> selected, selected_split, selected_branch;
        auto values = input;
        const char * names[] = {"cpu","npu","gpu"};
        for (int il = 0; il < 2; ++il) {
            auto & l = *layers[il];
            auto cpu_ref = l.base.run(values,"cpu_native",1);
            l.ids = cpu_ref.ids; l.counts = cpu_ref.counts;
            validate_values(cpu_ref.output,ref["layers"][std::to_string(il)].get<std::vector<float>>(),nt,"native_attention_reference");
            std::vector<int> sorted;
            for (int e = 0; e < ne; ++e) { if (l.counts[e]) { sorted.push_back(e); } }
            std::sort(sorted.begin(),sorted.end(),[&](int a,int b) { return l.counts[a] == l.counts[b] ? a < b : l.counts[a] < l.counts[b]; });
            if (!frozen.is_null()) {
                require(frozen.at("tokens").get<int>() == nt && frozen.at("prefix_length") == report["prefix_length"], "frozen workload mismatch");
                const auto & search = frozen.at("search").at(il);
                require(search.at("histogram").get<std::vector<int32_t>>() == l.counts, "frozen histogram mismatch");
                auto lookup = [&](const char * key) {
                    for (const auto & entry : search.at("candidates")) {
                        if (entry.at("name") == search.at(key)) {
                            return placement{entry.at("name").get<std::string>(),entry.at("owner").get<std::vector<int>>(),entry.at("shared").get<int>()};
                        }
                    }
                    throw std::runtime_error("missing frozen placement");
                };
                selected.push_back(lookup("winner")); selected_split.push_back(lookup("split_winner"));
                selected_branch.push_back(lookup("branch_winner"));
                report["search"].push_back(search); report["frozen_search"] = argv[5];
                values = cpu_ref.output; continue;
            }
            std::vector<placement> candidates;
            for (int main = 0; main < 3; ++main) {
                for (int shared = 0; shared < 3; ++shared) {
                    candidates.push_back({std::string(names[main])+"_shared_"+names[shared],std::vector<int>(ne,main),shared});
                }
            }
            for (int main : {0,1}) {
                for (int other = 0; other < 3; ++other) {
                    if (other == main) { continue; }
                    for (int count : {1,2,3,4,8,12,16,24,32}) {
                        if (count >= int(sorted.size())) { continue; }
                        for (int shared : {main,2}) {
                            if (main == 2 && shared == 2) { continue; }
                            placement p{std::string(names[main])+"_tail"+std::to_string(count)+"_"+names[other]+"_shared_"+names[shared],std::vector<int>(ne,main),shared};
                            for (int i = 0; i < count; ++i) { p.owner[sorted[i]] = other; }
                            candidates.push_back(std::move(p));
                        }
                    }
                }
            }
            double best = INFINITY, best_split = INFINITY, best_branch = INFINITY;
            placement winner, split_winner, branch_winner;
            json search = {{"layer",il},{"histogram",l.counts},{"candidates",json::array()}};
            for (const auto & p : candidates) {
                l.prepare(p);
                auto actual = l.run(values);
                auto check = chain_difference(actual,cpu_ref.output,nt);
                std::vector<double> times;
                for (int r = 0; r < 3; ++r) { auto start=timer::now(); l.run(values); times.push_back(elapsed(start)); }
                double median = stats(times,nt)["median_us"];
                search["candidates"].push_back({{"name",p.name},{"owner",p.owner},{"shared",p.shared},{"timing",stats(times,nt)},{"check",check}});
                fprintf(stderr,"SEARCH layer=%d %s %.3f ms valid=%d\n",il,p.name.c_str(),median/1000,check["passes_4percent_per_token"].get<bool>());
                if (check["passes_4percent_per_token"].get<bool>() && median < best) { best=median; winner=p; }
                bool split = p.name.find("_tail") != std::string::npos;
                if (check["passes_4percent_per_token"].get<bool>() && split && median < best_split) { best_split=median; split_winner=p; }
                if (check["passes_4percent_per_token"].get<bool>() && !split && median < best_branch) { best_branch=median; branch_winner=p; }
            }
            require(std::isfinite(best),"no valid placement"); selected.push_back(winner);
            selected_split.push_back(std::isfinite(best_split) ? split_winner : winner);
            selected_branch.push_back(branch_winner);
            search["winner"]=winner.name; search["split_winner"]=selected_split.back().name;
            search["branch_winner"]=branch_winner.name; report["search"].push_back(search); save();
            values = cpu_ref.output;
        }
        // Freeze the search winners, then run independent balanced measurement rounds.
        for (int depth : {1,2}) {
            json item = {{"layers",depth},{"samples",json::array()},{"validation",json::object()}};
            std::vector<double> times[6];
            for (int round = -2; round < reps; ++round) {
                for (int order = 0; order < 6; ++order) {
                    int p = ((round+2)+order)%6;
                    for (int il=0; il<depth; ++il) {
                        placement plan = p==3 ? selected[il] : p==4 ? selected_split[il] : p==5 ? selected_branch[il] :
                            placement{std::string(names[p])+"_all",std::vector<int>(ne,p),p};
                        layers[il]->prepare(plan);
                        // Initialization and backend graph preparation are excluded equally.
                        layers[il]->run(il ? ref["layers"]["0"].get<std::vector<float>>() : input);
                    }
                    auto start=timer::now(); auto result=input;
                    std::vector<json> phases(depth);
                    for(int il=0;il<depth;++il) { result=layers[il]->run(result,&phases[il]); }
                    double total=elapsed(start);
                    std::string name=p==3 ? "selected" : p==4 ? "split" : p==5 ? "branch" : names[p];
                    auto check=chain_difference(result,ref["layers"][std::to_string(depth-1)].get<std::vector<float>>(),nt);
                    item["validation"][name]=check;
                    require(check["passes_4percent_per_token"].get<bool>(), "confirmation numerical validation failed");
                    if(round>=0) {
                        times[p].push_back(total);
                        item["samples"].push_back({{"round",round},{"policy",name},{"total_us",total},{"phases",phases},{"check",check}});
                    }
                }
            }
            for(int p=0;p<6;++p) { item["summary"][p==3?"selected":p==4?"split":p==5?"branch":names[p]]=stats(times[p],nt); }
            report["confirmation"].push_back(item); save();
        }
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
#endif
