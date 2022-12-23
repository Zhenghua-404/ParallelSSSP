#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;


// a structure to represent a weighted edge in graph

struct Edge {
	int src, dest;
	int weight;
};


// a structure to represent a connected, directed and
// weighted graph
struct Graph {
	// V-> Number of vertices, E-> Number of edges
	int V, E;

	// graph is represented as an array of edges.
	struct Edge* edge;
};

// Creates a graph with V vertices and E edges
struct Graph* createGraph(int V, int E)
{
	struct Graph* graph = new Graph;
	graph->V = V;
	graph->E = E;
	graph->edge = new Edge[E];
	return graph;
}

struct Graph* generateEdges(int E, float density){
	int V = sqrt(E / density);
	int edgePerNode = (int)(density*V);
	srand(10);

	int actual_E=edgePerNode*V;
	struct Graph* graph = createGraph(V, actual_E);

	//connect each node to density*V other random nodes
	//total edges: density*V^2=density*E/density=E, correct
	for(int eachNode=0; eachNode<V; eachNode++){
		for(int e=0; e<edgePerNode; e++){
			int edgeId = eachNode*edgePerNode+e;
			graph->edge[edgeId].src=eachNode;
			int dest = rand()%V;
			while(dest==eachNode){
				dest = rand()%V;
			}
			graph->edge[edgeId].dest=dest;
			//random int from 0 to 9, inclusive.
			int weight = rand()%10;
			graph->edge[edgeId].weight=weight;
		}
	}

	//test
	printf("Graph V: %d\n", V);
	printf("Graph E: %d\n", E);
	printf("Graph actual E: %d\n", actual_E);
	printf("Graph density: %f\n", density);
	printf("Edge per node: %d\n", edgePerNode);
	printf("src -> dest: weight\n");
	//print first 10 edges
	/*
	for(int i=0; i<10; i++){
			if(i<actual_E){
				printf("%d -> %d: %d\n", graph->edge[i].src, graph->edge[i].dest, graph->edge[i].weight);		
			}
	}
	*/
    return graph;
}



// check runtime call error
#define cudaSafeCall(call) {  \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "cudaSafeCall: %s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

// check kernel launch error
#define cudaCheckErr(errorMessage) {    \
  cudaError_t err = cudaGetLastError(); \
  if(cudaSuccess != err){               \
    fprintf(stderr, "cudaError: %s(%i) : %s : %s.\n", __FILE__, __LINE__, errorMessage, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                 \
}}


// A utility function used to print the solution
void printArr(int dist[], int n)
{
	printf("10 Vertex Distance from Source\n");
	for (int i = 0; i < 10 && i<n; ++i)
		printf("%d \t\t %d\n", i, dist[i]);
}

__global__ void relax(struct Edge* edges, int E, int *dist){
    //each thread process each edge
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
    if (tid<E) {
			int u = edges[tid].src;
			int v = edges[tid].dest;
			int weight = edges[tid].weight;
			if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
	 				dist[v]=dist[u]+weight;
	 			//printf("tid %d has dist[%d]: %d\n", tid, v, dist[v]);
				//atomicExch(&(dist[v]), dist[u] + weight);
        //printf("tid %d updated dist[%d] to %d\n", tid, v, dist[u] + weight);
    }
}


__global__ void checkNegCycle(struct Edge* edges, int E, int *dist){
    //each thread process each vertex
    int tid = blockDim.x * blockIdx.x +threadIdx.x;
		if (tid<E) {
			int u = edges[tid].src;
			int v = edges[tid].dest;
			int weight = edges[tid].weight;
			if (dist[u] != INT_MAX
				&& dist[u] + weight < dist[v])
				printf("Graph contains negative weight cycle");
			  return; 
		}
}


// The main function that finds shortest distances from src
// to all other vertices using Bellman-Ford algorithm. The
// function also detects negative weight cycle
void BellmanFord_parallel(struct Graph* graph, int* dist, int src)
{
	int V = graph->V;
	int E = graph->E;
  //printf("Prallel begins: number of edges: %d\n", E);

	// Step 1: Initialize distances from src to all other
	// vertices as INFINITE
	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX;
	dist[src] = 0.0;
 
  // copy distance to device
  int* d_dist;
  cudaMalloc(&d_dist, V*sizeof(int));
  cudaMemcpy(d_dist, dist, V*sizeof(int), cudaMemcpyHostToDevice);
 
  // copy graph to device
  struct Edge *edges = (struct Edge*)malloc(sizeof(struct Edge)*E);
 for(int i=0; i<E; i++){
     edges[i]=graph->edge[i];
 }
	struct Edge* d_edges;
	cudaMalloc((void**)&d_edges, sizeof(struct Edge)*E);
	cudaMemcpy(d_edges, edges, sizeof(struct Edge)*E, cudaMemcpyHostToDevice);

 
  //printf("Allocated data...\n");
  
	// Step 2: Relax all edges |V| - 1 times. A simple
	// shortest path from src to any other vertex can have
	// at-most |V| - 1 edges
	//the outer loop has loop dependency: a->b->c depends on a->b
	//therefore parallelize the inner loop
  //printf("Number of blocks of threads: %d\n",1+(E-1)/1024);
	for (int i = 1; i <= V - 1; i++) {
      //1D? Do we need 2D or more?
      relax<<<1+(E-1)/1024, 1024>>>(d_edges, E, d_dist);
	}
  //printf("Relax function finished...\n");

	// Step 3: check for negative-weight cycles. The above
	// step guarantees shortest distances if graph doesn't
	// contain negative weight cycle. If we get a shorter
	// path, then there is a cycle.
	checkNegCycle<<<1+(E-1)/1024, 1024>>>(d_edges, E, d_dist);
  
  cudaSafeCall(cudaDeviceSynchronize());
  cudaCheckErr("kernel error");
  
  //printf("Error checked...\n");
 
 
  cudaMemcpy(dist, d_dist, V*sizeof(int), cudaMemcpyDeviceToHost);

	printArr(dist, V);
 
  free(edges);
  cudaFree(d_edges);
  cudaFree(d_dist);

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
	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX;
	dist[src] = 0;

	// Step 2: Relax all edges |V| - 1 times. A simple
	// shortest path from src to any other vertex can have
	// at-most |V| - 1 edges
	for (int i = 1; i <= V - 1; i++) {
		//read in parallel, write one by one
		for (int j = 0; j < E; j++) {
			int u = graph->edge[j].src;
			int v = graph->edge[j].dest;
			int weight = graph->edge[j].weight;
			if (dist[u] != INT_MAX
				&& dist[u] + weight < dist[v])
				dist[v] = dist[u] + weight;
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

void checkSum(int distS[], int distP[], int size){
	int count1=0;
	int count2=0;
	for(int i=0; i<size; i++){
		count1+=distS[i];
		count2+=distP[i];
	}
	printf("Sum of distances for Serial: %d.\nSum of distances for Parallel: %d.\n", count1, count2);
}

struct Graph* loadData(int E, float density, std::string filename){
		int V = sqrt(E / density);
		int edgePerNode = (int)(density*V);
		int actual_E=edgePerNode*V;
		struct Graph* graph = createGraph(V, actual_E);
		ifstream file;
		file.open(filename);
		
		string line;
		int row=0;
		while(getline(file, line))
		{
				if(row==0){
						row+=1;
						continue;
				}
			//split line into 3 fields
			std::string str = line;
			std::vector<int> vect;

			std::stringstream ss(str);

			for (int i; ss >> i;) {
					vect.push_back(i);    
					if (ss.peek() == ',')
							ss.ignore();
			}
			graph->edge[row-1].src=vect[0];
			graph->edge[row-1].dest=vect[1];
			graph->edge[row-1].weight=vect[2];
			row+=1;
			
		}
		
		file.close();
		return graph;
}


std::string to_string_with_precision(float a_value, int n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Driver's code
int main()
{
	// To test, change E and density

	int E = 1000; // Number of edges in graph
	float density = 0.8;
	int V = sqrt(E / density);

	int *distP= (int *)malloc(sizeof(int)*V);
	int *distS= (int *)malloc(sizeof(int)*V);
	struct Graph* graph = loadData(E, density,std::to_string(E)+"E"+to_string_with_precision(density,2)+"D"+".txt");

  int t = clock();
	BellmanFord_serial(graph, distS, 0);
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Bellman serial took %d ms\n", t);
  printf("---------------------------------\n");
  t = clock();
	BellmanFord_parallel(graph, distP, 0);
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Bellman parallel took %d ms\n", t);
	
  printf("---------------------------------\n");
	checkSum(distS, distP,V);
	

	free(distP);
	free(distS);
	return 0;
}
