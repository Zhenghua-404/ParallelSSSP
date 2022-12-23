//Create a random graph with n nodes and m edges
//Input: m, density (n would be decided by m and density)

#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include <sstream>


std::string to_string_with_precision(float a_value, int n)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


// a structure to represent a weighted edge in graph
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
	srand(1);

	int actual_E=edgePerNode*V;

	struct Graph* graph = createGraph(V, actual_E);

	//connect each node to density*V other random nodes
	//total edges: density*V^2=density*E/density=E, correct
	for(int eachNode=0; eachNode<V; eachNode++){
		//need to record end nodes to avoid duplicate!
		// initialize with -1: unassigned
		//int endNodes[]={-1};
		for(int e=0; e<edgePerNode; e++){
			int edgeId = eachNode*edgePerNode+e;
			graph->edge[edgeId].src=eachNode;
			int dest = rand()%V;
			//make sure end node is not equal to start node
			while(dest==eachNode){
				dest = rand()%V;
			}
			graph->edge[edgeId].dest=dest;
			//random int from 1 to 9, inclusive.
			int weight = (rand()%9)+1;
			graph->edge[edgeId].weight=weight;
		}
	}

	//test
//	printf("Graph V: %d\n", V);
//	printf("Graph E: %d\n", E);
//	printf("Graph actual E: %d\n", actual_E);
//	printf("Graph density: %f\n", density);
//	printf("Edge per node: %d\n", edgePerNode);
//	printf("src -> dest: weight\n");
	//print first 10 edges
	// Create and open a text file
	ofstream MyFile(std::to_string(E)+"E"+to_string_with_precision(density,2)+"D"+".txt");
	std::string header = std::to_string(E)+","+std::to_string(actual_E)+","+std::to_string(V)+","+std::to_string(density);
	MyFile << header << endl;

	for(int i=0; i<E; i++){\
		if(i<actual_E){
		//printf("%d -> %d: %d\n", graph->edge[i].src, graph->edge[i].dest, graph->edge[i].weight);
		std::string row= std::to_string(graph->edge[i].src)+","+std::to_string(graph->edge[i].dest)+","+std::to_string(graph->edge[i].weight);
		MyFile << row << endl;
		}
	}

	   //Close the file
	MyFile.close();
    return graph;
}



