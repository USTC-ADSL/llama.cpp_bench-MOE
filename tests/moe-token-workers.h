#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// One synchronous caller; workers sleep between jobs and never share a token.
// Backend completion precedes dispatch, and join precedes arena reuse.
class moe_token_workers {
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    std::function<void(int, int)> job_;
    std::exception_ptr error_;
    size_t generation_ = 0;
    int count_, tokens_ = 0, remaining_ = 0;
    bool stop_ = false;

    void worker(int id) {
        size_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) { return; }
            seen = generation_;
            int begin = tokens_ * id / count_, end = tokens_ * (id + 1) / count_;
            lock.unlock();
            std::exception_ptr error;
            try { job_(begin, end); } catch (...) { error = std::current_exception(); }
            lock.lock();
            if (error && !error_) { error_ = error; }
            if (--remaining_ == 0) { done_.notify_one(); }
        }
    }

public:
    explicit moe_token_workers(int count) : count_(count) {
        if (count < 2 || count > 8) { throw std::invalid_argument("host worker count must be 2..8"); }
        try {
            for (int id = 1; id < count; ++id) { workers_.emplace_back([this, id] { worker(id); }); }
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
            ready_.notify_all();
            for (auto & t : workers_) { t.join(); }
            throw;
        }
    }
    ~moe_token_workers() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto & t : workers_) { t.join(); }
    }
    int count() const { return count_; }
    void run(int tokens, const std::function<void(int, int)> & job) {
        if (tokens < count_) { job(0, tokens); return; }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = job; tokens_ = tokens; error_ = nullptr;
            remaining_ = count_ - 1; ++generation_;
        }
        ready_.notify_all();
        std::exception_ptr error;
        try { job(0, tokens / count_); } catch (...) { error = std::current_exception(); }
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        job_ = {};
        if (error) { std::rethrow_exception(error); }
        if (error_) { std::rethrow_exception(error_); }
    }
};
