/*Thread-local rand() and srand() for TrackerBench on Linux.

    TLD, Boosting and MIL in opencv_contrib draw their random features (and Boosting its online
    updates) from the C library's rand(). In glibc that is one generator shared by the whole
    process, so when markers are tracked on several threads (TrackerBench --jobs) each tracker
    would receive a different part of the sequence depending on thread timing, and results would
    change from run to run. These definitions take precedence over glibc's for the OpenCV
    libraries too (the executable is searched first when symbols are resolved), and give every
    thread its own generator. It is glibc's own algorithm and state size, so after srand(s) a
    thread produces exactly the sequence a fresh process produces after srand(s); a new thread
    starts as a new process does, as if seeded with 1.

    The Microsoft C runtime already keeps the rand() state per thread, so nothing is needed there.
    Only TrackerBench links this file; the GUI application tracks markers on one thread.*/

#include <cstdint>
#include <cstdlib>      // also defines __GLIBC__ on glibc systems, so it must come first

#if defined(__GLIBC__)

namespace
{
struct ThreadGenerator
{
    random_data data{};             // must be zeroed before initstate_r
    char state[128] = {};           // glibc's default state size (the TYPE_3 generator rand() uses)

    ThreadGenerator() { initstate_r(1, state, sizeof(state), &data); }
};

ThreadGenerator& Generator()
{
    thread_local ThreadGenerator generator;
    return generator;
}
}   // namespace

extern "C" int rand(void) noexcept
{
    int32_t value = 0;
    random_r(&Generator().data, &value);
    return value;
}

extern "C" void srand(unsigned int seed) noexcept
{
    srandom_r(seed, &Generator().data);
}

#endif
