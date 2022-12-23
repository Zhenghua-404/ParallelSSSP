#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <limits.h>
#include <set>

using  namespace std;
struct Dist {
	int id;    // node
	int dist;   // distance
	bool operator < (const Dist& a) const {
		return this->dist > a.dist;
	}
};

class Solution {
public:
	void getShortestDist(map<int, vector<Dist>>& graph, map<int, int>& dists, int start) {
		priority_queue<Dist> myQ;       // min heap builds a priority queue
		set<int> mySet;                // processed nodes
		Dist firstOne = { start, 0 };
		myQ.push(firstOne);

		while (!myQ.empty()) {
			Dist tmpD = myQ.top();
			myQ.pop();  // Delete minimum
			if (mySet.find(tmpD.id) == mySet.end()) {
				for (auto& d : graph[tmpD.id]) {
					if (mySet.find(d.id) != mySet.end()) {
						continue;
					}
					else {
						Dist nextD;
						nextD.id = d.id;
						nextD.dist = d.dist + tmpD.dist;
						myQ.push(nextD); // Insert the queue
					}
				}
				mySet.insert(tmpD.id);
				dists[tmpD.id] = tmpD.dist;
			}
			else {
				continue;
			}
		}
	}
};