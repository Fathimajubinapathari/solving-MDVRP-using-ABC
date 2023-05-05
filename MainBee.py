import time
import numpy as np
import matplotlib.pyplot as plt

import bee
import localsearch as ls
import datamapping as dm



def initParam(filename):
    # the data file
    raw_data = dm.Importer()
    raw_data.import_data(filename)

    # vehicle capacity
    capacity = int(raw_data.info["CAPACITY"])
    print(capacity)

    # depot
    depot = raw_data.depot
    depot = [i - 1 for i in depot]

    # node demand
    print(depot)
    demand_list = raw_data.demand_array
    for dep in depot:
        demand_list[dep] = 999999

    citylist = np.linspace(0, len(demand_list) - 1, len(demand_list))
    citylist_tabu = list(np.copy(citylist))
    node_num = len(citylist)

    # coordination
    coordination = raw_data.node_coordinates_list

    # distance and fitness
    distance_matrix = raw_data.distance_matrix
    fitness_matrix = []
    for row in distance_matrix:
        fitness_row = []
        for distance in row:
            if distance == 0:
                fitness = 0
            else:
                fitness = 1 / distance
            fitness_row.append(fitness)
        fitness_matrix.append(fitness_row)

    return capacity, depot, demand_list, citylist, citylist_tabu, node_num, coordination, distance_matrix, fitness_matrix


# plot the graph
def showResult(compare_set, coordination, depot):
    for i in compare_set:
        tour = compare_set[i]
        x = []
        y = []
        for j in tour:
            x.append(coordination[j][0])
            y.append(coordination[j][1])
            random_color = [i[0] for i in np.random.rand(3, 1)]
            plt.scatter(x, y, marker="*", color=random_color)
        plt.plot(x, y, c=random_color, label=f"Route{i+1}")
    z = []
    w = []
    for i in depot:
        z.append(coordination[i][0])
        w.append(coordination[i][1])
    for index in range(len(coordination)):
        plt.text(coordination[index][0], coordination[index][1], index + 1)
    plt.scatter(z, w, marker="s", color="red", label="Depot")
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))


    plt.xlabel('City x coordination')
    plt.ylabel("City y coordination")
    plt.title("The VRP map by Bee")
    plt.legend()
    plt.show()

start_time = time.time()
# -----------------Solve VRP using ABC-Meta-Heuristic-----------#
def beeHeuristic(capacity, depot, demand_list, citylist, citylist_tabu, node_num, coordination, distance_matrix, fitness_matrix):
    global compare_set
    iterations, population = 20, 80
    local_search, lamada, nn = "on", 1.2, 1
    compare_result = float('inf')
    final_result = float('inf')
    final_set = {}
    reverse_distance_matrix = fitness_matrix

    # employed bee phase
    for iter in range(iterations):
        '''run with multi replications to determine the iteration best result'''
        result_iter, tour_set_iter = bee.iteration(compare_result, depot, node_num, demand_list, capacity, citylist, citylist_tabu, distance_matrix, fitness_matrix, population, nn)

        if result_iter < compare_result:
            compare_result = result_iter
            compare_set = tour_set_iter
        else:
            pass
        print("iteration %i: " % iter, compare_result)

#    Scout bee phase
        localsearch_set = {}
        localsearch_result = 0
        if local_search == "on":
            for i in tour_set_iter:
                compare_tour = compare_set[i]
                length = len(compare_tour)
                improve = ls.TwoOptSwap(compare_tour, i, distance_matrix)
                localsearch_set[improve.result] = improve.tour
                localsearch_result += improve.result


    #Scout bee phase
        result_iter, tour_set_iter = bee.iteration(localsearch_result, depot, node_num, demand_list, capacity, citylist,
                                                   citylist_tabu, distance_matrix, fitness_matrix, population, nn)

        if result_iter < localsearch_result:
            localsearch_result = result_iter
            localsearch_set = tour_set_iter
        else:
            pass

        route_num = 0
        for distance, route in final_set.items():
            route_num += 1
        #print('Cost %s' % final_result)

        if localsearch_result < final_result:
            final_result = localsearch_result
            final_set = localsearch_set
        else:
            pass

        compare_result = 999999
        print(final_set)

    return final_result, final_set


if __name__ == '__main__':
    filename = "P01.vrp"
    capacity, depot, demand_list, citylist, citylist_tabu, node_num, coordination, distance_matrix, fitness_matrix = initParam(filename)
    final_result, final_set  = beeHeuristic(capacity, depot, demand_list, citylist, citylist_tabu, node_num, coordination, distance_matrix,fitness_matrix)

    print(final_set, 'final_set')
    print(final_result, 'Final distance')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    '''plot'''
    showResult(final_set, coordination, depot)
