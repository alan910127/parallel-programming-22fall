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

void init_distance(int* distances, int size) {
#pragma omp parallel for schedule(static, 16)
  for (int i = 0; i < size; ++i) {
    distances[i] = NOT_VISITED_MARKER;
  }

  distances[ROOT_NODE_ID] = 0;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(Graph g, int* distances, int current_level) {
  int next_level_count = 0;

  // process `current_level` in parallel
#pragma omp parallel for reduction(+ : next_level_count)
  for (int node = 0; node < g->num_nodes; node++) {
    // only consider the vertices that are on `current_level`
    if (distances[node] != current_level) continue;

    int start_edge = g->outgoing_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->outgoing_starts[node + 1];

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int outgoing = g->outgoing_edges[neighbor];

      if (distances[outgoing] != NOT_VISITED_MARKER) continue;

      distances[outgoing] = distances[node] + 1;
      next_level_count++;
    }
  }

  return next_level_count;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
  // explore the graph level-by-level, instead of vertex-by-vertex

  init_distance(sol->distances, graph->num_nodes);

  int current_level = 0;
  int next_level_count = 1;  // ROOT is on level 0

  while (next_level_count > 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    next_level_count = top_down_step(graph, sol->distances, current_level);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", next_level_count,
           end_time - start_time);
#endif

    // go on to the next level
    ++current_level;
  }
}

int bottom_up_step(Graph g, int* distances, int previous_level) {
  int current_level_count = 0;

  for (int node = 0; node < g->num_nodes; ++node) {
    if (distances[node] != NOT_VISITED_MARKER) continue;

    int start_edge = g->incoming_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->incoming_starts[node + 1];

    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int incoming = g->incoming_edges[neighbor];

      if (distances[incoming] != previous_level) continue;

      distances[node] = previous_level + 1;
      current_level_count++;
      break;
      // if an ancestor is found, do the step for another vertex
    }
  }

  return current_level_count;
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

  int current_level = 0;
  int next_level_count = 1;  // ROOT is on level 0

  while (next_level_count > 0) {
    next_level_count = bottom_up_step(graph, sol->distances, current_level);
    ++current_level;
  }
}

void bfs_hybrid(Graph graph, solution* sol) {
  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.

  init_distance(sol->distances, graph->num_nodes);

  int current_level = 0;
  int next_level_count = 1;  // ROOT is on level 0
}
