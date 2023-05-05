import numpy
from collections import deque

# this class importing the object
class Importer(object):
    '''Read the meta data from the file'''

    def __init__(self):
        self.file_lines = []
        self.info = {}
        self.node_coordinates_list = []
        self.distance_matrix = None
        self.demand_array = None
        self.depot = []

    def import_data(self, filename):
        self._read_file(filename)
        self.info, break_lines = self._read_info()
        self.node_coordinates_list, demand_list, self.depot = \
            self._return_nodes_and_delivery_lists(break_lines)
        adjacency_matrix_list = \
            self._create_distance_matrix(self.node_coordinates_list, int(self.info["DIMENSION"]))
        self.distance_matrix = numpy.array(adjacency_matrix_list)
        self.demand_array = numpy.array(demand_list)

    def _read_file(self, my_filename):
        filelines = []
        with open(my_filename, "rt") as f:
            filelines = f.read().splitlines()
        self.file_lines = filelines

    def _read_info(self):

        my_filelines = self.file_lines
        info = {}
        start = 0
        middle = 0
        end = 0

        for i, line in enumerate(my_filelines):
            if line.startswith("NODE_COORD_SECTION"):
                start = i
            elif line.startswith("DEMAND_SECTION"):
                middle = i
            elif line.startswith("DEPOT_SECTION"):
                end = i
            elif line.startswith("EOF"):
                break
            elif line.split(' ')[0].isupper():  # checks if line begins with UPPERCASE key
                splited = line.split(':')
                info[splited[0].strip()] = splited[1].strip()

        return info, (start, middle, end)

    def _return_nodes_and_delivery_lists(self, my_breaklines):
        my_filelines = self.file_lines
        start, middle, end = my_breaklines
        node_coordinates_list = []
        demand_list = []
        depot = []

        for i, line in enumerate(my_filelines):
            if start < i < middle:
                splited = line.strip().split(' ')
                splited = list(map(float, splited))
                node_coordinates_list.append((splited[1], splited[2]))

            if middle < i < end:
                splited = line.split(' ')
                splited = splited[:2]
                splited = list(map(int, splited))
                demand_list.append(splited[1])

            if i > end:
                splited = line.split(' ')
                if splited[0] == 'EOF':
                    break
                splited = splited[1]  # the data format has ' '

                if splited != '-1':
                    depot.append(int(splited))

        return node_coordinates_list, demand_list, depot

    # find the euclidian distance between all nodes
    def _euclidian_distance(self, my_node1, my_node2):
        x1, y1 = my_node1
        x2, y2 = my_node2

        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance

    # create the distance matrix by using euclidian distance
    def _create_distance_matrix(self, my_node_coordinates_list, my_dimension):
        ncl = deque(my_node_coordinates_list[:])
        matrix = []
        while ncl:
            row = [0] * (my_dimension + 1 - len(ncl))
            node1 = ncl.popleft()
            for node2 in ncl:
                row.append(self._euclidian_distance(node1, node2))
            matrix.append(row)

        # mirroring the matrix
        for i in range(my_dimension):
            for j in range(my_dimension):
                try:
                    matrix[j][i] = matrix[i][j]
                except IndexError as e:
                    print("##ERROR!##\nBad indexing: " + str((i, j)))
                    print("that definitly shouldnt happen, it >might< be a problem with the imported file")
                    raise e
        return matrix
