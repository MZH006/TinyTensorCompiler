#pragma once

#include <iosfwd>

class Graph;

// Emit a standalone program for one fused linear operation with runtime x
// and embedded finite weights/bias. The program reads raw float values on stdin.
void emitCpp(const Graph& graph, std::ostream& out);
