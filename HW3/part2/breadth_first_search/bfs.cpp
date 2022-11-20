#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstddef>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) { list->count = 0; }

void vertex_set_init(vertex_set* list, int count) {
  list->max_vertices = count;
  list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

static inline void init_distance(int* distances, int size) {
  // mark all nodes as unvisited
#pragma omp parallel for schedule(static, 16)
  for (int i = 0; i < size; ++i) {
    distances[i] = NOT_VISITED_MARKER;
  }

  // d(ROOT, ROOT) = 0
  distances[ROOT_NODE_ID] = 0;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(Graph g, int* distances, int current_level,
                  int* edges_from_frontier = nullptr) {
  int next_level_count = 0;
  int m_f = 0;

  // process `current_level` in parallel
#pragma omp parallel for reduction(+ : next_level_count, m_f)
  for (int i = 0; i < g->num_nodes; i++) {
    // only consider the vertices that are on `current_level` (i.e. frontier)
    if (distances[i] == current_level) {
      const Vertex* begin = outgoing_begin(g, i);
      const Vertex* end = outgoing_end(g, i);
      m_f += outgoing_size(g, i);

      // attempt to add all neighbors to the new frontier
      for (auto neighbor = begin; neighbor != end; ++neighbor) {
        int outgoing = *neighbor;

        if (distances[outgoing] == NOT_VISITED_MARKER) {
          distances[outgoing] = distances[i] + 1;
          next_level_count++;
        }
      }
    }
  }

  if (edges_from_frontier != nullptr) *edges_from_frontier = m_f;
  return next_level_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
  // explore the graph level-by-level, instead of vertex-by-vertex

  init_distance(sol->distances, graph->num_nodes);

  for (int next_level_count = 1, current_level = 0; next_level_count > 0;
       ++current_level) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    next_level_count = top_down_step(graph, sol->distances, current_level);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", next_level_count,
           end_time - start_time);
#endif
  }
}

int bottom_up_step(Graph g, int* distances, int current_level,
                   int* edges_from_unexplored = nullptr) {
  int next_level_count = 0;
  int m_u = 0;

#pragma omp parallel for reduction(+ : next_level_count, m_u)
  for (int i = 0; i < g->num_nodes; ++i) {
    if (distances[i] == NOT_VISITED_MARKER) {
      const Vertex* begin = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      m_u += incoming_size(g, i);

      for (auto neighbor = begin; neighbor != end; ++neighbor) {
        int incoming = *neighbor;

        if (distances[incoming] == current_level) {
          next_level_count++;
          distances[i] = distances[incoming] + 1;
          // if an ancestor is found, do the step for another vertex
          break;
        }
      }
    }
  }

  if (edges_from_unexplored != nullptr) *edges_from_unexplored = m_u;
  return next_level_count;
}

void bfs_bottom_up(Graph graph, solution* sol) {
  // For PP students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.

  init_distance(sol->distances, graph->num_nodes);

  for (int next_level_count = 1, current_level = 0; next_level_count > 0;
       ++current_level) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    next_level_count = bottom_up_step(graph, sol->distances, current_level);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", next_level_count,
           end_time - start_time);
#endif
  }
}

void bfs_hybrid(Graph graph, solution* sol) {
  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.

  static constexpr int ALPHA = 14;
  static constexpr int BETA = 24;

  int m_f = 0, m_u = 0;

  init_distance(sol->distances, graph->num_nodes);

  for (int next_level_count = 1, current_level = 0; next_level_count > 0;
       ++current_level) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    bool change_tb = m_f * ALPHA > m_u;
    bool change_bt = next_level_count * BETA < graph->num_nodes;

    if (change_tb && !change_bt) {
      next_level_count =
          bottom_up_step(graph, sol->distances, current_level, &m_u);

    } else {
      next_level_count =
          top_down_step(graph, sol->distances, current_level, &m_f);
    }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", next_level_count,
           end_time - start_time);
#endif
  }
}
