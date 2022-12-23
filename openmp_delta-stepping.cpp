#include <iostream>
#include<vector>
#include<omp.h>
#include <ctime>
#include "Dijkstra.cpp"
#include<stdlib.h>
using namespace std;

#define INF 1<<21
#define N 10000000
int n, m, maxbucket;//n is the number of nodes  m is the number of edges maxbucket is the max of B
struct edge { int node, weight; };
vector<edge> G[N];//delta stepping Graph
int delta; // delta
int dis[N];// shortest path distance
vector<int> B[N];//Bucket
struct req { int v, w; };
vector<req> Rl;// light weight
vector<req> Rh;//heavy weight
map<int, vector<Dist>> myGraph; // Dijkstra Graph
map<int, int> shortestDists;    // shortest distance


bool bempty() {
	//if all buckets are empty
	for (int i = 0; i <= maxbucket; i++)
		if (!B[i].empty())
			return false;//No
	return true;//yes
}

void relax(int w, int d) {
	//Relaxation function
	//w end node d new distance
	if (d < dis[w]) {
		// new distance is shorter than the old one
		if (dis[w] != INF) {
			vector<int>::iterator res = find(B[dis[w] / delta].begin(), B[dis[w] / delta].end(), w);
			if (res != B[dis[w] / delta].end())
#pragma omp critical (C)
				B[dis[w] / delta].erase(res);// Delete the old distance from bucket
		}
		if (d / delta > maxbucket) 
			maxbucket = d / delta;
#pragma opm critical (D)
		B[d / delta].push_back(w); //New distance is added to the bucket
		dis[w] = d; // Update the distance
	}
}

void deltastepping(int s) {
	maxbucket = 0;
	for (int i = 0; i < n; i++)
		dis[i] = INF;
	relax(s, 0);// add the source point to the bucket
	int j = 0;
	omp_set_num_threads(8);// three threads of three loops
	while (!bempty()) {
		Rl.clear();
		Rh.clear();
		while (!B[j].empty()) {
			
#pragma omp for private (k)
			// Build request lists
			for (int i = 0; i < B[j].size(); i++) {
				int vv = B[j][i];
				for (int k = 0; k < G[vv].size(); k++) {
					req r;
					r.v = G[vv][k].node;
					r.w = dis[vv] + G[vv][k].weight;

					if (G[vv][k].weight <= delta) {
#pragma omp critical (A)
						Rl.push_back(r);
					}
					else {
#pragma omp critical (B)
						Rh.push_back(r);
					}
				}
			}
			B[j].clear();
#pragma omp for
			// relax light edges
			for (int i = 0; i < Rl.size(); i++)
				relax(Rl[i].v, Rl[i].w);


			Rl.clear();
		}

#pragma omp for
		//relax heavy edges Only one time
		for (int i = 0; i < Rh.size(); i++)
			relax(Rh[i].v, Rh[i].w);
		Rh.clear();
		j++;
	}
}

int main() {

	srand((int)time(0));
	clock_t startTime1,startTime2, endTime;
	int total_edge = 0;
	float density = 0.0;
	int a, b, c;
	FILE* fp;
	errno_t err = fopen_s(&fp, "graphs/100000E0.80D.txt", "r");
	if (err)
		printf("Can't open the file! \n"); 
	else {
		fscanf_s(fp, "%d,%d,%d,%f", &total_edge, &m,&n, &density);
		for (int i = 0; i < m; i++) {
			fscanf_s(fp, "%d,%d,%d", &a, &b, &c);
			G[a].push_back({ b,c });
			myGraph[a].push_back({ b,c });
		}
	}

	delta = 50;


	cout << "The graph has " << n << " nodes" << " and " << m << " edges." << endl;


	

	//Dijkstra method
	Solution solver;
	startTime2 = clock();
	solver.getShortestDist(myGraph, shortestDists, 0);
	endTime = clock();
	cout << "The Dijkstra running time is " << (double)(endTime - startTime2) << " ms." << endl;

	//delta-stepping method
	startTime1 = clock();
	deltastepping(0);
	endTime = clock();
	cout << "The openmp delta-stepping running time is " << (double)(endTime - startTime1) << " ms." << endl;


	int sum = 0;
	int sum2 = 0;
	for (int i = 0; i < n; i++) {
		if (dis[i] != INF) {
			sum += dis[i];
		}
		sum2 += shortestDists[i];
	}
	printf("Serial Successful! Sum:%d\n", sum2);
	printf("Openmp Successful! Sum:%d\n", sum);
	

	return 0;
}
