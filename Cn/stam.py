class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + " connected to " + str([nbr.id for nbr in self.connectedTo])

    def get_connections(self):
        return self.connectedTo.keys()

    def get_id(self):
        return self.id

    def get_weights(self, nbr):
        return self.connectedTo[nbr]


class Graph:

    def __init__(self):
        self.vertices = {}
        self.n_vertices = 0

    def add_vertex(self, key):
        self.n_vertices += 1
        newVertex = Vertex(key)
        self.vertices[key] = newVertex
        return newVertex

    def get_vertex(self, key):
        if key in self.vertices:
            return self.vertices[key]
        else:
            return None

    def __contains__(self, key):
        return key in self.vertices

    def add_edge(self, f, t, w=0):
        if f not in self.vertices:
            self.add_vertex(f)
        if t not in self.vertices:
            self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], w)

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())

