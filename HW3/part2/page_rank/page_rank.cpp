#include "page_rank.h"

#include <omp.h>
#include <stdlib.h>

#include <cmath>
#include <memory>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence) {
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
#pragma omp parallel for schedule(static, 16)
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi])
     }; converged = (global_diff < convergence)
     }

   */

  auto score_new = std::make_unique<double[]>(numNodes);

  while (true) {
    // sum over all nodes vj reachable from incoming edges
#pragma omp parallel for schedule(static, 16)
    for (int vi = 0; vi < numNodes; ++vi) {
      auto incoming_edge_start = incoming_begin(g, vi);
      auto incoming_edge_end = incoming_end(g, vi);
      double sum = 0.0;
      for (auto vj = incoming_edge_start; vj != incoming_edge_end; ++vj) {
        sum = sum + solution[*vj] / outgoing_size(g, *vj);
      }
      score_new[vi] = sum;
    }

#pragma omp parallel for schedule(static, 16)
    for (int vi = 0; vi < numNodes; ++vi) {
      score_new[vi] = (damping * score_new[vi]) + (1.0 - damping) / numNodes;
    }

    // sum over all nodes v in graph with no outgoing edges
    double sum_isolated = 0.0;
#pragma omp parallel for reduction(+ : sum_isolated)
    for (int vi = 0; vi < numNodes; ++vi) {
      if (outgoing_size(g, vi) != 0) continue;
      sum_isolated = sum_isolated + damping * solution[vi] / numNodes;
    }

#pragma omp parallel for schedule(static, 16)
    for (int vi = 0; vi < numNodes; ++vi) {
      score_new[vi] = score_new[vi] + sum_isolated;
    }

    double global_diff = 0.0;
#pragma omp parallel for reduction(+ : global_diff)
    for (int vi = 0; vi < numNodes; ++vi) {
      global_diff = global_diff + abs(score_new[vi] - solution[vi]);
      solution[vi] = score_new[vi];
    }

    if (global_diff < convergence) break;
  }
}
