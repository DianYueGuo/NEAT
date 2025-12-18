#include <NEAT/connection.hpp>

using namespace neat;

Connection::Connection(int innovId, int inNodeId, int outNodeId, float weight, bool enabled): innovId(innovId), inNodeId(inNodeId), outNodeId(outNodeId), weight(weight), enabled(enabled) {
}
