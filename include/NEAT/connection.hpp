#ifndef CONNECTION_HPP
#define CONNECTION_HPP

namespace neat {

class Connection{
	public:
		int innovId;
		int inNodeId;
		int outNodeId;
		float weight;
		bool enabled;

		Connection(int innovId, int inNodeId, int outNodeId, float weight, bool enabled);
		Connection() {};
};

}

#endif	// CONNECTION_HPP
