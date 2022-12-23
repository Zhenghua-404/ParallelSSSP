#include <iostream>
#include<vector>
#include<omp.h>
#include <ctime>
#include <map>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

#define INF 1<<21
#define N 1000
#define N2 100000

struct Dist {
	int id;    // node
	int dist;   // distance
	bool operator < (const Dist& a) const {
		return this->dist > a.dist;
	}
};

#define CHK(call){                  \
cudaError_t err = call;             \
if (err != cudaSuccess) {           \
	printf(cudaGetErrorString(err)); \
	cudaDeviceReset();               \
	exit(1);                         \
}                                    \
}



struct Lock { //source: ¡°CUDA By Example¡± textbook
	int* mutex;
	Lock() { //constructor: create a Lock and initialize it to 0
		cudaMalloc(&mutex, sizeof(int));
		cudaMemset(mutex, 0, sizeof(int));
	}
	~Lock() { //destructor
		cudaFree(mutex);
	}
	__device__ void lock() {
		// if mutex == 0 set it to 1, otherwise don¡¯t change it. Then, return old value of mutex
		while (atomicCAS(mutex, 0, 1) != 0);
		__threadfence();
	}
	__device__ void unlock() {
		atomicExch(mutex, 0);
		__threadfence();
	}
};

int n, m;//n is the number of nodes  m is the number of edges
struct edge { int s_node,e_node, weight; };

edge* G;//delta stepping Graph
int delta; // delta
int *dis;// shortest path distance

struct req { int v, w; };

map<int, vector<Dist>> myGraph; // Dijkstra Graph
map<int, int> shortestDists;    // shortest distance

__device__ void req_push(req* R, req *r, int* length) {
	// Add new edge into list
	R[*length].v = r[0].v;
	R[*length].w = r[0].w;
	*length = *length + 1;
}

bool bempty(int Bucket[N][N]) {
	//if all buckets are empty
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (Bucket[i][j] != -1)
				return false;//No
		}
	}
	return true;//yes
}

bool empty(int* Bucket) {
	//if all are empty
	for (int i = 0; i < N; i++) {
		if (Bucket[i] != -1)
			return false;//No
	}
	return true;//yes
}


__global__ void relax(req* R, int* R_length, int* dis, int* S_R, int* flag) {
	//Parallel Relax function
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < *R_length) {
		
		int w = R[idx].v;
		int d = R[idx].w;
		//return idx of edge needs to relax
		if (d < dis[w]) {

			S_R[idx] = idx;
		}
	}
}

void S_relax(int w, int d,int *dis,int B[N][N],int delta) {
	//Serial Relaxation function
	//w end node, d new distance
	if (d < dis[w]) {
		// new distance is shorter than the old one
		if (dis[w] != INF) {
			for (int i = 0; i < N; i++) {
				// Delete the old distance from bucket
				if (B[dis[w] / delta][i] == w) {
					B[dis[w] / delta][i] = -1;
					break;
				}
			}
		}
		//New distance is added to the bucket
		for (int j = 0; j < N; j++) {
			if (B[d/delta][j] == -1) {
				B[d/delta][j] = w;
				break;
			}
		} 
		dis[w] = d; // Update the distance
	}
}

__global__ void request(int* B, edge* d_G_2d,int m, int* dis, req* d_Rl, req* d_Rh, int delta, int B_length, int* Rl_length, int* Rh_length,Lock *lock) {

	//Parallel Build Request List
	int threadID = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx < B_length) {
	
		if (B[idx] != -1) {
			int vv = B[idx];//vertex in the Bucket
			for (int i = 0; i < m; i++) {
				if (d_G_2d[i].s_node == vv) {
					//Divid all the edges of this vertex into light list and heavy list
					req r;
					r.v = d_G_2d[i].e_node;
					r.w = dis[vv] + d_G_2d[i].weight;
					if (d_G_2d[i].weight <= delta) {
						lock[0].lock();
						req_push(d_Rl, &r, Rl_length);
						lock[0].unlock();
					}
					else {
						lock[1].lock();
						req_push(d_Rh, &r, Rh_length);
						lock[1].unlock();
					}
				}
				
			}
		}

	}
}

int main() {

	srand((int)time(0));
	clock_t startTime1, startTime2, endTime;
	
	int total_edge = 0;
	float density = 0.0;
	int a, b, c;
	//open file and read the graph
	FILE* fp;
	errno_t err = fopen_s(&fp, "graphs/100000E0.80D.txt", "r");
	if (err)
		printf("Can't open the file! \n");
	else {
		fscanf_s(fp, "%d,%d,%d,%f", &total_edge, &m, &n, &density);
		G = (edge*)malloc(sizeof(edge) * m);
		dis = (int*)malloc(sizeof(int) * n);
		for (int i = 0; i < m; i++) {
			fscanf_s(fp, "%d,%d,%d", &a, &b, &c);
			G[i].s_node = a;
			G[i].e_node = b;
			G[i].weight = c;
		}
	}

	
	delta = 50;
	int *Rl_length =(int*)malloc(sizeof(int));// the length of light edges list
	*Rl_length = 0;
	int *Rh_length = (int*)malloc(sizeof(int));// the length of heavy edges list
	*Rh_length = 0;
	int Bucket[N][N];//Bucket
	memset(Bucket, -1, sizeof(Bucket));

	cout << "The graph has " << n << " nodes" << " and " << m << " edges." << endl;

	maxbucket = 0;
	for (int i = 0; i < n; i++)
		dis[i] = INF;
	Bucket[0][0] = 0;


	dis[0] = 0; //add the source point to the bucket

	edge* d_G_2d;//copy graph to GPU
	CHK(cudaMalloc((void**)&d_G_2d, sizeof(edge) * m));
	CHK(cudaMemcpy(d_G_2d, G, sizeof(edge) * m, cudaMemcpyHostToDevice)); 


	//int d_delta;
	int* d_dis;
	CHK(cudaMalloc((void**)&d_dis, sizeof(int) * n));
	CHK(cudaMemcpy(d_dis, dis, sizeof(int) * n, cudaMemcpyHostToDevice));

	//copy bucket to GPU

	int* d_Bi;

	CHK(cudaMalloc((void**)&d_Bi, sizeof(int) * N));

	
	int* d_B;//bucket
	CHK(cudaMalloc((void**)&d_B, sizeof(int) * N * N));
	
	size_t d_pitch;
	CHK(cudaMallocPitch((void**)&d_B, &d_pitch, sizeof(int) * N, N));

	
	CHK(cudaMemcpy2D(
		d_B,    
		d_pitch,    
		Bucket,    
		sizeof(int) * N,    
		sizeof(int) * N,    
		N,    
		cudaMemcpyHostToDevice    
	));


	req* d_Rl;// light weight
	req* d_Rh;//heavy weight
	req* C_Rl = (req*)malloc(N2 * sizeof(req));// light weight
	req* C_Rh = (req*)malloc(N2 * sizeof(req));//heavy weight

	CHK(cudaMalloc((void**)&d_Rl, sizeof(req) * N2));
	CHK(cudaMalloc((void**)&d_Rh, sizeof(req) * N2));
	cudaMemcpy(d_Rl, C_Rl, sizeof(req) * N2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rh, C_Rh, sizeof(req) * N2, cudaMemcpyHostToDevice);


	int* d_Rl_length, * d_Rh_length;
	cudaMalloc((void**)&d_Rl_length, sizeof(int));
	cudaMalloc((void**)&d_Rh_length, sizeof(int));

	int* flag = (int *)malloc(sizeof(int));
	*flag = 0;
	int* d_flag;
	cudaMalloc(&d_flag, sizeof(int));
	cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);

	Lock lock[2];
	Lock* d_lock;

	CHK(cudaMalloc((void**)&d_lock, 2 * sizeof(Lock)));
	CHK(cudaMemcpy(d_lock, lock,2 * sizeof(Lock),cudaMemcpyHostToDevice));


	int* d_S_Rl;// light weight
	int* d_S_Rh;//heavy weight

	int* S_Rl = (int*)malloc(N2 * sizeof(int));// need to Serial relax light list
	int* S_Rh = (int*)malloc(N2 * sizeof(int));//need to Serial relax heavy list

	memset(S_Rl, -1, sizeof(int)*N2);
	memset(S_Rh, -1, sizeof(int)*N2);

	CHK(cudaMalloc((void**)&d_S_Rl, sizeof(int) * N2));
	CHK(cudaMalloc((void**)&d_S_Rh, sizeof(int) * N2));
	cudaMemcpy(d_S_Rl, S_Rl, sizeof(int) * N2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S_Rh, S_Rh, sizeof(int) * N2, cudaMemcpyHostToDevice);


	int nThreads = 1024;
	int nBlocks = 0;

	startTime1 = clock();

	int j = 0;
	while (!bempty(Bucket)) {
		*Rl_length = 0;//empty the light list and heavy list
		*Rh_length = 0;
		CHK(cudaMemcpy(d_Rh_length, Rh_length, sizeof(int), cudaMemcpyHostToDevice));

		while (empty(Bucket[j]) != true) {
			CHK(cudaMemcpy(d_Rl_length, Rl_length, sizeof(int), cudaMemcpyHostToDevice));
			CHK(cudaMemcpy(d_Bi, &Bucket[j][0], sizeof(int) * N, cudaMemcpyHostToDevice));

			// Build request lists
			nBlocks = (N - 1) / nThreads + 1;
			

			request << <nBlocks, nThreads >> > (d_Bi, d_G_2d, m, d_dis, d_Rl, d_Rh, delta, N, d_Rl_length, d_Rh_length,d_lock);


			CHK(cudaGetLastError()); //1
			CHK(cudaDeviceSynchronize());

			

			cudaMemcpy(Rl_length, d_Rl_length, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Rh_length, d_Rh_length, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(C_Rl, d_Rl, sizeof(req)*(N2), cudaMemcpyDeviceToHost);
			cudaMemcpy(C_Rh, d_Rh, sizeof(req)*(N2), cudaMemcpyDeviceToHost);

			//Empty Bucket[j][]
			for (int i = 0; i < N; i++) {
				Bucket[j][i] = -1;
			}
			

			//Copy the Bucket from CPU to GPU
			cudaMemcpy2D(
				d_B,   
				d_pitch,  
				Bucket,   
				sizeof(int) * N,   
				sizeof(int) * N,    
				N,   
				cudaMemcpyHostToDevice    
			);

			
			nBlocks = (*Rl_length - 1) / nThreads + 1;

			//Find the edges in light list need to relax parallelly
			relax << <nBlocks, nThreads >> > (d_Rl, d_Rl_length, d_dis, d_S_Rl, d_flag);
		
			CHK(cudaGetLastError()); //1
			CHK(cudaDeviceSynchronize());
			cudaMemcpy(flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

			cudaMemcpy(S_Rl, d_S_Rl, sizeof(int)*N2, cudaMemcpyDeviceToHost);

			//Relax Serially
			for (int i = 0; i < N2; i++) {
				if (S_Rl[i] != -1) {
					int idx = S_Rl[i];
					S_relax(C_Rl[idx].v, C_Rl[idx].w, dis, Bucket, delta);
				}
			}

			//Copy dis[] and bucket to GPU
			CHK(cudaMemcpy(d_dis, dis, sizeof(int) * n, cudaMemcpyHostToDevice));

			
			cudaMemcpy2D(
				d_B,   
				d_pitch,    
				Bucket,    
				sizeof(int) * N,    
				sizeof(int) * N,    
				N,   
				cudaMemcpyHostToDevice    
			);

			*Rl_length = 0;
			memset(S_Rl, -1, sizeof(int));
			cudaMemcpy(d_Rl_length, Rl_length, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_S_Rl, S_Rl, sizeof(int)*N2, cudaMemcpyHostToDevice);

		}


		//Find the edges in heavy list need to relax parallelly
		nBlocks = (*Rh_length - 1) / nThreads + 1;
		relax << <nBlocks, nThreads >> > (d_Rh, d_Rh_length, d_dis, d_S_Rh, d_flag);

		CHK(cudaGetLastError()); //1
		CHK(cudaDeviceSynchronize());

		cudaMemcpy(S_Rh, d_S_Rh, sizeof(int) * N2, cudaMemcpyDeviceToHost);

		//Relax Serially
		for (int i = 0; i < N2; i++) {
			if (S_Rh[i] != -1) {
				int idx = S_Rh[i];
				S_relax(C_Rh[idx].v, C_Rh[idx].w, dis, Bucket, delta);
			}
		}

		CHK(cudaMemcpy(d_dis, dis, sizeof(int) * n, cudaMemcpyHostToDevice));

		
		cudaMemcpy2D(
			d_B,    
			d_pitch,    
			Bucket,    
			sizeof(int) * N,    
			sizeof(int) * N,    
			N,    
			cudaMemcpyHostToDevice    
		);

		*Rh_length = 0;
		memset(S_Rh, -1, sizeof(int));
		cudaMemcpy(d_Rh_length, Rh_length, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_S_Rh, S_Rh, sizeof(int) * N2, cudaMemcpyHostToDevice);

		//Rh.clear();

		j++;
	}

	endTime = clock();
	cout << "The CUDA delta-stepping running time is " << (double)(endTime - startTime1) << " ms." << endl;

	//Thu sum of the path
	int sum = 0;
	for (int i = 0; i < n; i++) {
		if (dis[i] != INF) {
			sum += dis[i];
		}
	}
	printf("CUDA Successful! Sum:%d\n", sum);
	//Free memory
	cudaFree(d_G_2d);
	cudaFree(d_dis);
	cudaFree(d_Bi);
	cudaFree(d_B);
	cudaFree(d_Rl); cudaFree(d_Rl_length);
	cudaFree(d_Rh); cudaFree(d_Rh_length);
	cudaFree(d_lock);
	cudaFree(d_S_Rl);
	cudaFree(d_S_Rh);
	return 0;
}
