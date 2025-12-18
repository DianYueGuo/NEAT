#include <NEAT/genome.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace neat;

Genome::Genome(int nbInput, int nbOutput, int nbHiddenInit, float probConnInit, std::vector<std::vector<int>>* innovIds, int* lastInnovId, float weightExtremumInit): weightExtremumInit(weightExtremumInit), nbInput(nbInput), nbOutput(nbOutput){
	speciesId = -1;
	// Initial hidden nodes and probabilistic connectivity are ignored for canonical NEAT initialization.
	(void) nbHiddenInit;
	(void) probConnInit;
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
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
		}
	}
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
	/* Process all sumInput and sumOutput. For that, it "scans" each layer from the inputs to the last hidden's layer to calculate sumInput with already known value. */ 
	
	// reset sums for non-input nodes
	for (int i = nbInput + 1; i < (int) nodes.size(); i++) {
		nodes[i].sumInput = 0;
		nodes[i].sumOutput = 0;
	}

	int lastLayer = 0;
	for (const auto& n : nodes) {
		if (n.layer > lastLayer) {
			lastLayer = n.layer;
		}
	}

	for (int ilayer = 0; ilayer < lastLayer; ilayer++) {
		// process sumInput
		for (int i = 0; i < (int) connections.size(); i++) {
			if (!connections[i].enabled) continue;	// connection disabled

			int inLayer = nodes[connections[i].inNodeId].layer;
			int outLayer = nodes[connections[i].outNodeId].layer;

			if (inLayer != ilayer) continue;	// only process nodes in current layer
			if (outLayer <= inLayer) continue;	// skip non-forward links

			nodes[connections[i].outNodeId].sumInput += nodes[connections[i].inNodeId].sumOutput * connections[i].weight;
		}
		
		// process sumOutput
		for (int i = 1 + nbInput; i < (int) nodes.size(); i++) {	// for each node except inputs because sumOutput has already be calculated
			if (nodes[i].layer == ilayer + 1) {
				nodes[i].sumOutput = activationFn(nodes[i].sumInput);
			}
		}
	}
}


void Genome::getOutputs(float outputs[]) {
	for (int i = 0; i < nbOutput; i++) {
		outputs[i] = nodes[1 + nbInput + i].sumOutput;
	}
}


void Genome::mutate(std::vector<std::vector<int>>* innovIds, int* lastInnovId, bool areRecurrentConnectionsAllowed, float mutateWeightThresh, float mutateWeightFullChangeThresh, float mutateWeightFactor, float addConnectionThresh, int maxIterationsFindConnectionThresh, float reactivateConnectionThresh, float addNodeThresh, int maxIterationsFindNodeThresh) {
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
		addConnection(innovIds, lastInnovId, maxIterationsFindConnectionThresh, areRecurrentConnectionsAllowed, reactivateConnectionThresh);
	}
	
	// ### NODES ###
	randomNb = (float) rand() / (float) RAND_MAX;
	while (randomNb < 1.0f + 1e-10 && randomNb > 1.0f - 1e-10) {	// == 1
		randomNb = (float) rand() / (float) RAND_MAX;
	}	// generate a random value in [0,1)
	if (randomNb < addNodeThresh) {
		// adding a node
		addNode(innovIds, lastInnovId, maxIterationsFindNodeThresh, areRecurrentConnectionsAllowed);
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
			// perturb weight additively (closer to original NEAT small jitter)
			connections[i].weight += ((float) rand() / (float) RAND_MAX * 2 * mutateWeightFactor - mutateWeightFactor);
		}
	}
}

bool Genome::addConnection(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindConnectionThresh, bool areRecurrentConnectionsAllowed, float reactivateConnectionThresh) {	// return true if the process ended well, false in the other case
	// find valid node pair
	int iterationNb = 0;
	int isValid = 0;
	int inNodeId = rand() % (int) nodes.size();
	int outNodeId = rand() % (int) nodes.size();
	while (iterationNb < maxIterationsFindConnectionThresh && isValid == 0) {
		inNodeId = rand() % (int) nodes.size();
		outNodeId = rand() % (int) nodes.size();
		isValid = isValidNewConnection(inNodeId, outNodeId, areRecurrentConnectionsAllowed);
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
			bool recurrent = false;
			if (isValid == 3) {
				recurrent = true;
			}
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, recurrent));
			return true;
		}
	} else {
		return false;	// cannot find a valid connection
	}
}

int Genome::isValidNewConnection(int inNodeId, int outNodeId, bool areRecurrentConnectionsAllowed) {	// 0 = not valid connection, 1 = valid connection, 2 = connection currently disabled, 3 = recurrent connections
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

bool Genome::addNode(std::vector<std::vector<int>>* innovIds, int* lastInnovId, int maxIterationsFindNodeThresh, bool areRecurrentConnectionsAllowed) {	// return true = node created, false = nothing created
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
			float weight = connections[iConn].weight;
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			
			// build second connection
			inNodeId = newNodeId;
			outNodeId = connections[iConn].outNodeId;
			innovId = getInnovId(innovIds, lastInnovId, inNodeId, outNodeId);
			weight = (float) rand() / (float) RAND_MAX * 2 * weightExtremumInit - weightExtremumInit;	// random number in [-weightExtremumInit; weightExtremumInit]
			connections.push_back(Connection(innovId, inNodeId, outNodeId, weight, true, false));
			
			// update layers
			nodes[newNodeId].layer = nodes[connections[iConn].inNodeId].layer + 1;	// update newNodeId layer
			nodes[connections[iConn].outNodeId].layer = nodes[newNodeId].layer + 1;	// update outNodeId layer
			updateLayersRec(connections[iConn].outNodeId);	// recursively update layers
			
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
            if (!connections[iConn].isRecurrent && connections[iConn].enabled && connections[iConn].inNodeId == current) {
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
    }
}
