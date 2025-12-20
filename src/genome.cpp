#include <NEAT/genome.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace neat;

Genome::Genome(int nbInput, int nbOutput, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit): weightExtremumInit(weightExtremumInit), nbInput(nbInput), nbOutput(nbOutput){
	speciesId = -1;
	// NODES
	// bias
	nodes.push_back(Node(0, 0));
	nodes[0].sumInput = 1;	// init value of the bias node
	nodes[0].sumOutput = 1;	// bias output must also start at 1 to propagate
	// input
	for (int i = 1; i < nbInput + 1; i++) {
		nodes.push_back(Node(i, 0));
	}
	// output (canonical NEAT starts fully connected input->output, no initial hidden)
	for (int i = nbInput + 1; i < nbInput + 1 + nbOutput; i++) {
		nodes.push_back(Node(i, 1));
	}
	
	// CONNECTIONS: fully connect inputs (and bias) to outputs initially
	for (int inNodeId = 0; inNodeId < nbInput + 1; inNodeId++) {
			for (int outNodeId = nbInput + 1; outNodeId < nbInput + 1 + nbOutput; outNodeId++) {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true));
			}
	}
	topoDirty = true;
	}

int Genome::getInnovId(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int inNodeId, int outNodeId) {
	/* get the innovation id of the connection inNodeId -> outNodeId. Create one if needed */
	if ((int) innovIds->size() < inNodeId + 1) {
		int previousSize = (int) innovIds->size();
		for (int k = 0; k < inNodeId + 1 - previousSize; k++) {	// we complete the array until we reach inNodeId
			innovIds->push_back({-1});
		}
	}
	if ((int) (*innovIds)[inNodeId].size() < outNodeId + 1) {
		int previousSize = (int) (*innovIds)[inNodeId].size();
		for (int k = 0; k < outNodeId + 1 - previousSize; k++) {	// we complete the array until we reach outNodeId
			(*innovIds)[inNodeId].push_back(-1);
		}
	}
	if ((*innovIds)[inNodeId][outNodeId] == -1) {	// has been this connection build in the past ?
		*lastInnovId += 1;
		(*innovIds)[inNodeId][outNodeId] = *lastInnovId;
	}
	
	return (*innovIds)[inNodeId][outNodeId];
}


void Genome::loadInputs(float inputs[]) {
	for (int i = 0; i < nbInput; i++) {
		nodes[i + 1].sumInput = inputs[i];	// i + 1 because the first node is the bias one
		nodes[i + 1].sumOutput = inputs[i];	// sumInput = sumOutput for the inputs nodes
	}
}


void Genome::runNetwork(float activationFn(float input)) {
	if (topoDirty) {
		rebuildTopology();
	}
	// reset sums for non-input nodes
	for (int i = nbInput + 1; i < (int) nodes.size(); i++) {
		nodes[i].sumInput = 0;
		nodes[i].sumOutput = 0;
	}

	// forward pass using cached topological order and adjacency
	for (int nodeId : topoOrder) {
		if (nodeId > nbInput) {
			nodes[nodeId].sumOutput = activationFn(nodes[nodeId].sumInput);
		}
		for (int edgeIdx : forwardAdj[nodeId]) {
			const Connection& conn = connections[edgeIdx];
			nodes[conn.outNodeId].sumInput += nodes[conn.inNodeId].sumOutput * conn.weight;
		}
	}
}


void Genome::getOutputs(float outputs[]) {
	for (int i = 0; i < nbOutput; i++) {
		outputs[i] = nodes[1 + nbInput + i].sumOutput;
	}
}


void Genome::mutate(std::vector<std::vector<int>>* innovIds, int* lastInnovId, float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh) {
	// ### WEIGHTS ###
	float randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)

	// mutating weights
	mutateWeights(mutateWeightFullChangeThresh, mutateWeightFactor, mutateWeightThresh);
	
	// ### CONNECTIONS ###
	randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < addConnectionThresh) {
		// adding a conection
		addConnection(innovIds, lastInnovId, maxIterationsFindConnectionThresh, reactivateConnectionThresh);
	}
	
	// ### NODES ###
	randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < addNodeThresh) {
		// adding a node
		addNode(innovIds, lastInnovId, maxIterationsFindNodeThresh);
	}
}

void Genome::mutateWeights(float mutateWeightFullChangeThresh, float mutateWeightFactor, float mutateWeightThresh) {
	for (int i = 0; i < (int) connections.size(); i++) {
		float randomNb = (float) rand() / (float) RAND_MAX;
		if (randomNb > mutateWeightThresh) {
			continue;
		}

		randomNb = (float) rand() / (float) RAND_MAX;
		while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
			randomNb = (float) rand() / (float) RAND_MAX;
		}	// generate a random value in [0,1)
		if (randomNb < mutateWeightFullChangeThresh) {
			// reset weight
			connections[i].weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;
		} else {
			// perturb weight with small Gaussian-ish jitter
			float u1 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
			float u2 = ((float) rand() + 1.0f) / ((float) RAND_MAX + 2.0f);
			float mag = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.1415926f * u2);
			connections[i].weight += mag * mutateWeightFactor;
		}
	}
}

bool Genome::addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
	// find valid node pair
	int iterationNb = 0;
	int isValid = 0;
	int inNodeId = rand() % (int) nodes.size();
	int outNodeId = rand() % (int) nodes.size();
	while (iterationNb < maxIterationsFindConnectionThresh && isValid == 0) {
		inNodeId = rand() % (int) nodes.size();
		outNodeId = rand() % (int) nodes.size();
		isValid = isValidNewConnection(inNodeId, outNodeId);
		iterationNb++;
	}
	
	if (iterationNb < maxIterationsFindConnectionThresh) {	// a valid connection has been found
		// mutating
		if (isValid == 2) {	// it is a former connection
			float randomNb = (float) rand() / (float) RAND_MAX;
			while ((int) randomNb == 1) {
				randomNb = (float) rand() / (float) RAND_MAX;
			}	// generate a random value in [0,1)
			if (randomNb < reactivateConnectionThresh) {
				// search the connection
				int iConnDisabled = 0;
				while (iConnDisabled < (int) connections.size() && !(!connections[iConnDisabled].enabled && connections[iConnDisabled].inNodeId == inNodeId && connections[iConnDisabled].outNodeId == outNodeId)) {
					iConnDisabled ++;
				}
				if (iConnDisabled < (int) connections.size()) {
					connections[iConnDisabled].enabled = true;	// former connection is reactivaded
					topoDirty = true;
					return true;
				} else {
					std::cout << "Error : Genome::addConnection" << std::endl;	// impossible
					return false;
				}
				} else {
					return true;	// return true even no connection has been change because process ended well
				}
			} else {
				int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true));
				topoDirty = true;
				return true;
			}
		} else {
			return false;	// cannot find a valid connection
		}
}

int Genome::isValidNewConnection(int inNodeId, int outNodeId) {	// 0 = not valid connection, 1 = valid connection, 2 = connection currently disabled
	if (inNodeId == outNodeId) return 0;	// the connection boucle itself
	if (nodes[inNodeId].layer == nodes[outNodeId].layer) return 0;	// the connection link to nodes on the same layer
	for (int i = 0; i < (int) connections.size(); i++) {
		if (connections[i].inNodeId == inNodeId && connections[i].outNodeId == outNodeId) {
			if (connections[i].enabled) {
				return 0;	// it is already a connection enabled
			} else {
				return 2;	// it is a former connection
			}
		}
	}
	if (nodes[inNodeId].layer > nodes[outNodeId].layer)	{
		return 0;	// recurrent connection are disallowed for canonical NEAT
	}
	return 1;	// test done : it is a valid connection !
}

bool Genome::addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindNodeThresh) {	// return true = node created, false = nothing created
	// choose at random an enabled forward connection
	if ((int) connections.size() > 0) {
		int iConn = rand() % (int) connections.size();
		int iterationNb = 0;
		while (iterationNb < maxIterationsFindNodeThresh && !(connections[iConn].enabled && nodes[connections[iConn].inNodeId].layer < nodes[connections[iConn].outNodeId].layer)) {
			iConn = rand() % (int) connections.size();
			iterationNb ++;
		}
		if (iterationNb < maxIterationsFindNodeThresh) {
			// disable former connection
			connections[iConn].enabled = false;
			
			// setup new node
			int newNodeId = (int) nodes.size();
			nodes.push_back(Node(newNodeId, -1));	// no layer for the moment
			
			// build first connection
			int inNodeId = connections[iConn].inNodeId;
			int outNodeId = newNodeId;
			int innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				connections.push_back(Connection(innovId, inNodeId, outNodeId, 1.0f, true));	// NEAT sets this to 1.0
				
				// build second connection
				inNodeId = newNodeId;
				outNodeId = connections[iConn].outNodeId;
			innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
				float weight = connections[iConn].weight;	// preserve original weight on second link
				connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true));
				
				// update layers
				nodes[newNodeId].layer = nodes[connections[iConn].inNodeId].layer + 1;	// update newNodeId layer
				int downstreamId = connections[iConn].outNodeId;
				int desiredLayer = nodes[newNodeId].layer + 1;
				// never shrink downstream layers; only lift if needed
				nodes[downstreamId].layer = std::max(nodes[downstreamId].layer, desiredLayer);
				updateLayersRec(downstreamId);	// recursively update layers
				topoDirty = true;
			
	return true;
		} else {
			std::cout << "Error : no active connection found in Genome::addNode" << std::endl;
			return false;	// no active connection found
		}
	} else {
		return false;	// there is no connection, cannot add a node
	}
}

void Genome::updateLayersRec(int nodeId) {
	// Avoid infinite recursion on cycles by iteratively propagating layer updates only when they increase.
    std::vector<int> stack;
    stack.push_back(nodeId);

    while (!stack.empty()) {
        int current = stack.back();
        stack.pop_back();
		if (current < 0 || current >= static_cast<int>(nodes.size())) {
            continue;
        }
        int baseLayer = nodes[current].layer;

        for (int iConn = 0; iConn < (int) connections.size(); iConn++) {
            if (connections[iConn].enabled && connections[iConn].inNodeId == current) {
                int newNodeId = connections[iConn].outNodeId;
				if (newNodeId < 0 || newNodeId >= static_cast<int>(nodes.size())) {
                    continue;
                }
                int newLayer = baseLayer + 1;
                if (nodes[newNodeId].layer < newLayer) {
                    nodes[newNodeId].layer = newLayer;
                    stack.push_back(newNodeId);
	        }
	    }
}

void Genome::ensureForwardLayers() {
	// Raise downstream layers until every enabled edge is forward.
	bool changed = true;
	int guard = static_cast<int>(nodes.size()) * 2 + static_cast<int>(connections.size());
	while (changed && guard-- > 0) {
		changed = false;
		for (const auto& conn : connections) {
			if (!conn.enabled) continue;
			if (conn.inNodeId < 0 || conn.inNodeId >= static_cast<int>(nodes.size())) continue;
			if (conn.outNodeId < 0 || conn.outNodeId >= static_cast<int>(nodes.size())) continue;
			int neededLayer = nodes[conn.inNodeId].layer + 1;
			if (nodes[conn.outNodeId].layer < neededLayer) {
				nodes[conn.outNodeId].layer = neededLayer;
				changed = true;
			}
		}
	}
}

void Genome::rebuildTopology() {
	ensureForwardLayers();
	forwardAdj.assign(nodes.size(), {});
	std::vector<int> indeg(nodes.size(), 0);
	for (int i = 0; i < (int) connections.size(); i++) {
		const auto& conn = connections[i];
		if (!conn.enabled) continue;
		if (conn.inNodeId < 0 || conn.inNodeId >= static_cast<int>(nodes.size())) continue;
		if (conn.outNodeId < 0 || conn.outNodeId >= static_cast<int>(nodes.size())) continue;
		if (nodes[conn.inNodeId].layer >= nodes[conn.outNodeId].layer) continue;	// skip non-forward
		forwardAdj[conn.inNodeId].push_back(i);
		indeg[conn.outNodeId]++;
	}
	topoOrder.clear();
	std::vector<int> queue;
	queue.reserve(nodes.size());
	for (int i = 0; i < (int) nodes.size(); i++) {
		if (indeg[i] == 0) queue.push_back(i);
	}
	for (size_t idx = 0; idx < queue.size(); idx++) {
		int nodeId = queue[idx];
		topoOrder.push_back(nodeId);
		for (int edgeIdx : forwardAdj[nodeId]) {
			int out = connections[edgeIdx].outNodeId;
			if (--indeg[out] == 0) queue.push_back(out);
		}
	}
	// if not all nodes reached (cycle), fall back to linear order as a guard
	if (topoOrder.size() != nodes.size()) {
		topoOrder.clear();
		for (int i = 0; i < (int) nodes.size(); i++) topoOrder.push_back(i);
	}
	topoDirty = false;
}
    }
}
