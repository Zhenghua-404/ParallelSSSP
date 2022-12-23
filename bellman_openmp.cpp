//============================================================================
// Name        : bellman_openmp.cpp
// Author      : mavis chen
// Configuration: Settings -> C compiler: gcc; C++ compiler: g++; add -fopenmp
//============================================================================

// A C++ program for Bellman-Ford's single source
// shortest path algorithm.
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;

//from graphGenerator.cpp import
struct Graph* generateEdges(int E, float density);


struct Edge {
	int src, dest, weight;
};

// a structure to represent a connected, directed and
// weighted graph
struct Graph {
	// V-> Number of vertices, E-> Number of edges
	int V, E;

	// graph is represented as an array of edges.
	struct Edge* edge;
};

// A utility function used to print the first 10(or n) results
void printArr(int *dist, int n)
{
	printf("10 Vertex Distance from Source\n");
	for (int i = 0; i < n && i<10; ++i)
		printf("%d \t\t %d\n", i, dist[i]);
}

// The main function that finds shortest distances from src
// to all other vertices using Bellman-Ford algorithm. The
// function also detects negative weight cycle
void BellmanFord_parallel(struct Graph* graph, int* dist, int src)
{
	int V = graph->V;
	int E = graph->E;

	// Step 1: Initialize distances from src to all other
	// vertices as INFINITE
	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX;
	dist[src] = 0;

	// Step 2: Relax all edges |V| - 1 times. A simple
	// shortest path from src to any other vertex can have
	// at-most |V| - 1 edges

	//the outer loop has loop dependency: a->b->c depends on a->b
	//therefore parallelize the inner loop
	printf("Set max threads: %d\n", omp_get_max_threads());
	for (int i = 1; i <= V - 1; i++) {
		#pragma omp parallel for num_threads(8)
		//read in parallel, write one by one
		for (int j = 0; j < E; j++) {
			int u = graph->edge[j].src;
			int v = graph->edge[j].dest;
			int weight = graph->edge[j].weight;
			if (dist[u] != INT_MAX
				&& dist[u] + weight < dist[v])
				{
				dist[v] = dist[u] + weight;
				//printf("thread %d set the distance of vertex %d to %d\n",omp_get_thread_num(),v,dist[v]);
				}

		}

	}

	printArr(dist, V);

	return;
}

// The main function that finds shortest distances from src
// to all other vertices using Bellman-Ford algorithm. The
// function also detects negative weight cycle
void BellmanFord_serial(struct Graph* graph, int* dist, int src)
{
	int V = graph->V;
	int E = graph->E;

	// Step 1: Initialize distances from src to all other
	// vertices as INFINITE
	for (int i = 0; i < V; i++){
		dist[i] = INT_MAX;
	}
	dist[src] = 0;

	// Step 2: Relax all edges |V| - 1 times. A simple
	// shortest path from src to any other vertex can have
	// at-most |V| - 1 edges
	for (int i = 1; i <= V - 1; i++) {
		//printf("Iteration %d\n", i);
		//read in parallel, write one by one
		for (int j = 0; j < E; j++) {
			int u = graph->edge[j].src;
			int v = graph->edge[j].dest;
			int weight = graph->edge[j].weight;
			if (dist[u] != INT_MAX
				&& dist[u] + weight < dist[v]){
				//printf("Update node %d from value %d to %d\n", v, dist[v], dist[u]+weight);
				dist[v] = dist[u] + weight;
			}

		}

	}

	// Step 3: check for negative-weight cycles. The above
	// step guarantees shortest distances if graph doesn't
	// contain negative weight cycle. If we get a shorter
	// path, then there is a cycle.
	for (int i = 0; i < E; i++) {
		int u = graph->edge[i].src;
		int v = graph->edge[i].dest;
		int weight = graph->edge[i].weight;
		if (dist[u] != INT_MAX
			&& dist[u] + weight < dist[v]) {
			printf("Graph contains negative weight cycle");
			return; // If negative cycle is detected, simply
					// return
		}
	}
	printArr(dist, V);
	return;
}

bool checkEqual(int* distS, int* distP, int size){
	int total =0;
	for(int i=0; i<size; i++){
		if(distS[i]!=distP[i]){
			printf("A parallel distance doesn't match at node %d:\nparallel dist: %d; serial dist: %d\n", i, distP[i], distS[i]);
			return false;
		}else{
			total+=distS[i];
		}
	}
	printf("Sum of dist for both algorithms: %d\n", total);
	return true;
}

// Driver's code
int main()
{

	int E = 2000000; // Number of edges in graph
	float density = 0.2;
	int V = sqrt(E / density);

	int *distP= (int *)malloc(sizeof(int)*V);
	int *distS= (int *)malloc(sizeof(int)*V);
	struct Graph* graph = generateEdges(E, density);

	// Function calldouble start;
	double start, end;
	start = omp_get_wtime();
	BellmanFord_serial(graph, distS, 0);
	end = omp_get_wtime();
	printf("Bellman serial took %f seconds\n", end - start);
	printf("---------------------------------\n");


	start = omp_get_wtime();
	BellmanFord_parallel(graph, distP, 0);
	end = omp_get_wtime();
	printf("Bellman Parallel took %f seconds\n", end - start);
	printf("---------------------------------\n");

	if(checkEqual(distS, distP,V)){
		printf("All parallel results match with serial results.\n");
	}

	free(distP);
	free(distS);
	return 0;
}
