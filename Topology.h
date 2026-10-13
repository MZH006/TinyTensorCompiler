#pragma once

#include <cstddef>

// Conservative invalidation across graphs, including edits through raw pointers.
// Graph construction, mutation, and execution are single-threaded.
namespace topology {
inline std::size_t revision = 0;
inline void changed() { ++revision; }
}
