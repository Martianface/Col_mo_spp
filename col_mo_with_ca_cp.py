# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 08:13:37 2014 V0.1

@author: martianface

Multi-agent systems with varying velocities (migrating from Matlab codes)

improvements:
1. running is fast without recording the distance matrix
2. simulation is executed with more agents
3. initial configuration is saved in files for later usage
4. arrows is plotted with python-based arrow functions
5. the histogram and time series of order parameter are plotted to determine the type of phase transition
*6. trying to use Monte Carlo simulation near the critical point
7. the absolute velocity of particles is calculated according to the distance between particles

updating history:

Updated on Sun Oct 05 13:45:08 2014 V0.2
1. The interaction graph of first layer neighbor is derived by r-limited Delaunay graph.
2. The threshold of setting absolute velocity to zero is set to 1e-13
3. Moving direction of particles is updated before moving.
Originally, vec_v = tmp_v * [cos(theta), sim(theta)], theta is the moving direction in the previous time.
Here, we use: vec_v = tmp_v * [cos(tmp_theta), sin(tmp_theta)] to include the effect of potential forces.
2. Minor modifications are used to optimize efficiency.
3. The threshold of minimum absolute velocity is set to 1e-12
4. The absolute velocity used to plan the moving direction can be chosen as follow:
   1) the value of previous time;
   2) the value of current planned absolute velocity;
   3) the value of unit vector, i.e. v_i(t) = 1;
   4) the value of consistent vector, such as v_i(t) = v_max
Different selections lead to backward update(BU) or forward update(FU). Here, we used FU strategy.

Updated on Tue Nov 04 10:38:08 2014
1. The r-limited Delaunay graph is obtained by a new designed function first_layer_neighbor(), in which the
   curve edges of the corresponding voronoi cell are replaced by the segment linking the two intersection
   points of two circles of the sensing range.
2. At given steps, the graphs of first layer neighbor are plotted by the old function first_layer_neighbor().
3. The dictionary data type in function are replaced by the list data type to increase the running speed.
4. Particles collisions and disconnection originate from the inaccuracy of float number in the following:
   1) The inaccuracy of floating number in calculating x, y components of the absolute velocity will lead to
      particles collision and disconnection. So The threshold of setting absolute velocity to zero is set
      to 1e-12 and when the norm of position offset is larger than planned absolute velocity, the correction
      process of the decomposition of velocity vectors is invoked.
   2) The tolerance for the equality of two float numbers leads to the misjudgement of first layer neighbors.
      The selection of tolerance will be considered to be adaptive for the different configurations.

Updated on Fri Nov 07 15:26:31
1. In the function limited_delaunay_neighbor(), we add the process of judging the distances between two
   consecutive vertices and two centers, respectively in order to exclude edges which are the approximation
   of arcs of voronoi cells.

Updated on Tue Nov 11 16:01:02
1. saving intermediate sensing range figures by replacing the approximating straight edges of voronoi cells
   by arcs.
2. new function new_limited_delaunay_neighbor() to obtain first layer neighbors according to the position_a,
   positions_b and voronoi_cell[a] is used.

Updated on Mon Dec 1 08:36:05
1. modifying the truncating strategy of position updating to avoid the inaccuracy of floating number. Setting
   a threshold to restrict the updating of positions when the norm of positions offset is less than 1e-11.

Updated on Sat Dec 13 17:14:02
1. modifying the strategy to avoid the inaccuracy of floating numbers by adding the distance between two
   particles a small increment when it is less than a threshold 1e-10.
2. the potential function of linear form is used and normalized to unify the coefficient of potential functions.

Modified on Wed Jan 14 2015 16:36:15
1. using the new defined function bisector() to replace function circle_intersection().
2. the parameter related to the phase transition is the magnitude of potential forces.

Updated on Thur Apr 2 01:03:55
1. optimizing the performance of the code according to suggestions from:
    http://www.ibm.com/developerworks/cn/linux/l-cn-python-optim/

Updated on Mon Apr 6 23:26:55
1. treating the problem with the tolerance of the minimum velocity :
    if the velocity is less than the V_TOL and the moving direction will
    degrade of the distance between two limited particles, then the velocity
    is set to ZERO,
    otherwise, the velocity is preserved to gradually improve the limited
    particles.

"""

import math
import random
import os
import time
import shutil  # file copy
import cPickle  # file saving as python object

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from descartes import PolygonPatch


# print the starting time
start_time = time.time()
print 'Starting: ' + time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
rs = 235013  # same random seed for all the parameter ua
random.seed(rs)  # with the same seed, we can obtain the same random numbers
# defining parameters and constants
N = 100  # the number of particles
SENSING_RANGE = math.sqrt(1)  # the sensing range of particle
CORE_RANGE = 0.01  # the radius of hard core of particle
NSTEPS = 100000  # the number of total steps of simulation
V_MAX = 0.03  # the max moving velocity
# the number of vertices of polygons approximating curves, only used to
# plot the fln graph.
RES = 50
PI = math.pi  # pi value
ZERO = 1e-4  # tolerance for checking two points coincidence
V_TOL = 1e-11  # tolerance for the absolute velocity
PS_TOL = 1e-14  # tolerance for the positions offset
ORIGIN = np.array([0, 0])  # original point in 2D
# time interval for checking the symmetry and connectivity
TEST_INTERVAL = 1
SAVE_INTERVAL = 20000  # time interval for saving the intermediate results

# defining the ini variables
neighbor_num = [0] * N  # the number of neighbor of a particle
neighbor_set = [0] * N  # the list recording the indices of neighbor particle
positions = np.array([[0.0, 0.0]] * N)  # positions of particles
u_m = 1.0  # user defined magnitude coefficient for regions of potential forces
rep_margin = 2 * CORE_RANGE + u_m * 2 * V_MAX
att_margin = SENSING_RANGE - u_m * 2 * V_MAX
u_a = 0.01  # magnitude of potential forces
u_b = 1e0  # magnitude of alignment forces
upsilon = 1e-3  # tolerance for radius of particle's polar coordinate


# user-defined functions used in the main function

def is_equal(a, b):
    '''
    judging the equality of floating numbers
    :para a b: two floating number
    :rtype: boolean True or False
    '''
    return abs(a - b) < ZERO


def two_points_distance(p1, p2):
    """
    calculating the distance between two points
    :param: p1, p2: 2d position coordinates of two points
    :rtype: float distance
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_on_segment(p1, p2, p3):
    """
    determining whether one polygon's vertex is
    on the edge of the other polygon
    :param: p1, p2, p3: p1 is the point of one polygon
                      p2 and p3 are points of another
                      polygon, two of which form an edge
    :rtype: boolean True or False
    """

    # whether p1 coincides with p2 or p3
    if is_equal(two_points_distance(p1, p2), 0) or is_equal(two_points_distance(p1, p3), 0):
        return True
    # p1 is in the region consisting of p2 and p3
    elif (min(p2[0], p3[0]) < p1[0] and p1[0] < max(p2[0], p3[0])) and (min(p2[1], p3[1]) < p1[1] and p1[1] < max(p2[1], p3[1])):
        # area of the triangle consisting of three points
        s = 0.5 * \
            ((p1[0] - p3[0]) * (p2[1] - p3[1]) -
             (p2[0] - p3[0]) * (p1[1] - p3[1]))
        if is_equal(s, 0):
            return True
        else:
            return False
    else:
        return False


def two_parallel_segments_colinearity(p1, p2, p3, p4):
    if not point_on_segment(p1, p3, p4) and \
       not point_on_segment(p2, p3, p4) and \
       not point_on_segment(p3, p1, p2) and \
       not point_on_segment(p4, p1, p2):
        return False
    else:
        return True


def two_points_slope(p1, p2):
    """
    calculating the slope of two points
    :param: p1, p2: 2d coordinates of two points
    :rtype: float slope
    """
    if abs(p1[0] - p2[0]) < 1e-15:
        return float('inf')  # degeneracy of infinite slope
    else:
        return (p2[1] - p1[1]) / (p2[0] - p1[0])


def bisector(p1, p2, r=SENSING_RANGE):
    """
    :param: p1,p2: the coordination of two particles, p1 is the center point,p2 is the neighboring point.
            r: the half sensing range of particles (the default value is r/2)
    :rtype: ip_sa, ip_ea: starting and ending angle of two intersection points between the bisector and circle.
            ip_sp, ip_ep: starting and ending point of two intersection points between the bisector and circle.
    """
    #  calculating the angle of neighboring point in the polar coordinate of
    #  the center point belonging to [0,2pi]
    # angle rotating from x axis to p2
    theta = math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))
    if theta < 0:
        theta += 2 * math.pi
    d = two_points_distance(p1, p2)
    delta_theta = math.acos(d / r)
    ip_sa = (theta + delta_theta) % (2 * math.pi)
    ip_ea = (theta - delta_theta) % (2 * math.pi)
    # global coordinates
    ip_sp = [(r / 2.0) * math.cos(ip_sa), (r / 2.0) * math.sin(ip_sa)] + p1
    ip_ep = [(r / 2.0) * math.cos(ip_ea), (r / 2.0) * math.sin(ip_ea)] + p1
    return ip_sp, ip_ep, ip_sa, ip_ea


def constrained_sensing_xrange(positions, r, res):
    """
    plotting the constrained sensing range of voronoi-like region
    :para positions(3*2 matrix): the first line is the coordinates of particle, and the remaining lines are the
            coordinates of two intersection points of sensing range circles between particle and its neighbors
          r: SENSING_RANGE
          res: the resolution of polygon approximating curve
    :rtype: approx_positions: N*2 array for the vertices of constrained sensing range polygon
    """
    # calculating theta1 belonging to [0,2pi]
    # first quadrant
    if positions[1][0] - positions[0][0] > 0 and positions[1][1] - positions[0][1] > 0:
        theta1 = math.atan2(
            positions[1][1] - positions[0][1], positions[1][0] - positions[0][0])
    # second quadrant
    elif positions[1][0] - positions[0][0] < 0 and positions[1][1] - positions[0][1] > 0:
        theta1 = math.atan2(
            positions[1][1] - positions[0][1], positions[1][0] - positions[0][0])
    # third quadrant
    elif positions[1][0] - positions[0][0] < 0 and positions[1][1] - positions[0][1] < 0:
        theta1 = math.atan2(
            positions[1][1] - positions[0][1], positions[1][0] - positions[0][0]) + 2 * PI
    else:  # forth quadrant
        theta1 = math.atan2(
            positions[1][1] - positions[0][1], positions[1][0] - positions[0][0]) + 2 * PI
    # calculating theta2 belonging to [0,2pi]
    # first quadrant
    if positions[2][0] - positions[0][0] > 0 and positions[2][1] - positions[0][1] > 0:
        theta2 = math.atan2(
            positions[2][1] - positions[0][1], positions[2][0] - positions[0][0])
    # second quadrant
    elif positions[2][0] - positions[0][0] < 0 and positions[2][1] - positions[0][1] > 0:
        theta2 = math.atan2(
            positions[2][1] - positions[0][1], positions[2][0] - positions[0][0])
    # third quadrant
    elif positions[2][0] - positions[0][0] < 0 and positions[2][1] - positions[0][1] < 0:
        theta2 = math.atan2(
            positions[2][1] - positions[0][1], positions[2][0] - positions[0][0]) + 2 * PI
    else:  # forth quadrant
        theta2 = math.atan2(
            positions[2][1] - positions[0][1], positions[2][0] - positions[0][0]) + 2 * PI
    # plotting the sector with the polygon by approximation (properly setting the starting and ending angle)
    # the center angle of sector is equal to 360 - gamma, which is calculated
    # according to cosine law
    a = math.sqrt((positions[0][0] - positions[2][0])
                  ** 2 + (positions[0][1] - positions[2][1]) ** 2)
    b = math.sqrt((positions[0][0] - positions[1][0])
                  ** 2 + (positions[0][1] - positions[1][1]) ** 2)
    c = math.sqrt((positions[1][0] - positions[2][0])
                  ** 2 + (positions[1][1] - positions[2][1]) ** 2)
    gamma = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    delta_theta = 2 * PI - gamma  # the rotating angle of the sector
    # starting and ending points are properly set
    theta_starting = theta1
    theta_ending = theta2
    # converting radian to degree for judging rotating angle of curve
    if not is_equal(abs((math.degrees(theta_starting) + math.degrees(delta_theta)) % 360), math.degrees(theta2)):
        theta_starting = theta2
        theta_ending = theta1

    # approximating sector with polygon
    del_alpha = delta_theta / float(res - 1)  # incremental angle
    starting_point = [positions[0][
        0] + r * math.cos(theta_starting), positions[0][1] + r * math.sin(theta_starting)]
    ending_point = [positions[0][
        0] + r * math.cos(theta_ending), positions[0][1] + r * math.sin(theta_ending)]
    interpolation_points = []
    # the first intersection point of two circles
    interpolation_points.append(starting_point)
    # the interpolation points between the two intersection points
    for i in xrange(1, res - 1):
        interpolation_points.append([positions[0][0] + r * math.cos(
            theta_starting + i * del_alpha), positions[0][1] + r * math.sin(theta_starting + i * del_alpha)])
    # the last intersection point of two circles
    interpolation_points.append(ending_point)
    approx_positions = np.asarray(interpolation_points)
    return approx_positions


def find_major_arc(positions, r):
    """
    according to center of circle, and intersection points of two circles, find the starting and ending point of a
    major arc corresponding to two intersection points.
    :param: positions(3*2 matrix): the first row is the coordinates of particle(center coordinates of circle), and
            the remaining two rows are the coordinates of two intersection points of two sensing range circles
            between particles and its nearest neighbors
            r: SENSING_RANGE/2
    :rtype: coordinates of starting and ending points
    """
    # calculating theta1 belonging to [0,2pi]
    theta1 = math.atan2(
        positions[1][1] - positions[0][1], positions[1][0] - positions[0][0])
    if theta1 < 0:
        theta1 += 2 * math.pi
    # more pythonic way: theta1 += 2*math.pi if theta1 < 0 else theta1

    theta2 = math.atan2(
        positions[2][1] - positions[0][1], positions[2][0] - positions[0][0])
    if theta2 < 0:
        theta2 += 2 * math.pi
    # plotting the sector with the polygon by approximation (properly setting the starting and ending angle)
    # the center angle of sector is equal to 360 - gamma, which is calculated
    # according to cosine law
    a = math.sqrt((positions[0][0] - positions[2][0])
                  ** 2 + (positions[0][1] - positions[2][1]) ** 2)
    b = math.sqrt((positions[0][0] - positions[1][0])
                  ** 2 + (positions[0][1] - positions[1][1]) ** 2)
    c = math.sqrt((positions[1][0] - positions[2][0])
                  ** 2 + (positions[1][1] - positions[2][1]) ** 2)
    gamma = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    delta_theta = 2 * math.pi - gamma  # the rotating angle of the sector

    # starting and ending points are represented by the global coordinates
    # with origin at positions[0] and angle rotates with ccw direction
    starting_angle = theta1
    starting_point = positions[1]
    ending_angle = theta2
    ending_point = positions[2]
    # converting radian to degree for judging rotating angle of curve
    if not is_equal(abs((math.degrees(starting_angle) + math.degrees(delta_theta)) % 360), math.degrees(ending_angle)):
        starting_angle = theta2
        starting_point = positions[2]
        ending_angle = theta1
        ending_point = positions[1]

    # all the info are respecting to the global coordinates, corresponding
    # angle should be calculated after minus the
    return starting_point, ending_point, starting_angle, ending_angle


def fill_circular_sector(starting_angle, ending_angle, center, r, ax, fcolor):
    """
    fill a circular sector with random color
    :param: starting_angle, ending_angle -- scalar represented by radian,
            angles are denoted by the local coordinate, whose origin is center
            center -- the center coordinates of the circle corresponded to the circular sector, 2d vector
            r -- radius of the circular sector
    """

    res = 50  # the number of interpolating points on the curve
    poly_circular_sector = []
    # starting angle in the first quadrant
    if starting_angle >= 0 and starting_angle < math.pi / 2:
        # ending angle in the first quadrant
        if ending_angle >= 0 and ending_angle < math.pi / 2:
            if ending_angle < starting_angle:
                theta = ending_angle - starting_angle
            else:
                theta = ending_angle - starting_angle - 2 * math.pi
        elif ending_angle >= math.pi / 2 and ending_angle < math.pi:
            theta = ending_angle - starting_angle - 2 * math.pi
        elif ending_angle >= math.pi and ending_angle < 3 * math.pi / 2:
            theta = ending_angle - starting_angle - 2 * math.pi
        else:
            theta = ending_angle - starting_angle - 2 * math.pi
    # starting angle in the second quadrant
    elif starting_angle >= math.pi / 2 and starting_angle < math.pi:
        # ending angle in the first quadrant
        if ending_angle >= 0 and ending_angle < math.pi / 2:
            theta = ending_angle - starting_angle
        elif ending_angle >= math.pi / 2 and ending_angle < math.pi:
            if ending_angle < starting_angle:
                theta = ending_angle - starting_angle
            else:
                theta = ending_angle - starting_angle - 2 * math.pi
        elif ending_angle >= math.pi and ending_angle < 3 * math.pi / 2:
            theta = ending_angle - starting_angle - 2 * math.pi
        else:
            theta = ending_angle - starting_angle - 2 * math.pi
    # starting angle in the third quadrant
    elif starting_angle >= math.pi and starting_angle < 3 * math.pi / 2:
        # ending angle in the first quadrant
        if ending_angle >= 0 and ending_angle < math.pi / 2:
            theta = ending_angle - starting_angle
        elif ending_angle >= math.pi / 2 and ending_angle < math.pi:
            theta = ending_angle - starting_angle
        elif ending_angle >= math.pi and ending_angle < 3 * math.pi / 2:
            if ending_angle < starting_angle:
                theta = ending_angle - starting_angle
            else:
                theta = ending_angle - starting_angle - 2 * math.pi
        else:
            theta = ending_angle - starting_angle - 2 * math.pi
    else:  # starting angle in the forth quadrant
        # ending angle in the first quadrant
        if ending_angle >= 0 and ending_angle < math.pi / 2:
            theta = ending_angle - starting_angle
        elif ending_angle >= math.pi / 2 and ending_angle < math.pi:
            theta = ending_angle - starting_angle
        elif ending_angle >= math.pi and ending_angle < 3 * math.pi / 2:
            theta = ending_angle - starting_angle
        else:
            if ending_angle < starting_angle:
                theta = ending_angle - starting_angle
            else:
                theta = ending_angle - starting_angle - 2 * math.pi

    # approximating sector with polygon
    del_theta = theta / float(res - 1)  # incremental angle
    # when the center angle of circular sector is less than pi, the center of
    # circle is added to the polygon
    poly_circular_sector.append(center)
    for i in xrange(res):
        poly_circular_sector.append(
            [center[0] + r * math.cos(starting_angle + i * del_theta), center[1] + r * math.sin(starting_angle + i * del_theta)])
    # fcolor = np.random.rand(3,1) # setting the color for filling the vn
    # region of particle
    a = Polygon(poly_circular_sector)
    patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
    ax.add_patch(patch)


def fill_triangle(positions, ax, fcolor):
    """
    plotting an triangle according to the info of vertices.
    :param: positions -- 3*2 matrix representing the three vertices of an triangle
            ax -- related to plotting the graph
            fcolor -- the color for filling the triangle
    """
    a = Polygon(positions)
    patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
    ax.add_patch(patch)


def limited_delaunay_neighbor(vc_a, position_a, position_b):
    """
    2014.11.17 modified func of limited_delaunay_neighbor() to avoid misjudgment of first layer neighbors according to the distance between two particles positions and two vertices of voronoi cells, the first layer neighbor of particle i is obtained.
    :param: vc_a -- the voronoi cell of particle i, shapely object
            position_a -- the coordinates of particle i
            position_b -- the coordinates of particle j
    :rtype: True -- the particle j is the first layer neighbor of particle i
            False -- otherwise
    """

    if vc_a.geom_type == 'LineString':
        point = list(vc_a.coords)
        # comparing the distance from the vertices of polygon to the particles'
        # position, respectively.
        d_v1_a = two_points_distance(point[0], position_a)
        d_v1_b = two_points_distance(point[0], position_b)
        d_v2_a = two_points_distance(point[1], position_a)
        d_v2_b = two_points_distance(point[1], position_b)
        # excluding the edge that is orthogonal to the segment between two
        # particles but not intersection segment of two voronoi cell
        if abs(d_v1_a - d_v1_b) < 1e-10 and abs(d_v2_a - d_v2_b) < 1e-10:
            return True
    else:
        # poly_a is a closed polygon, so the last vertex is the same as the
        # first point
        poly_a = np.array(vc_a.exterior)
        # the number of vertices in poly_a plus 1, since the last vertex is
        # repeated as the first one for a closed polygon
        poly_len = len(poly_a) - 1
        # the last point of polygons is repeated the first one
        for i in xrange(poly_len):
            # excluding the exceptional vertices formed by numerical inaccuracy
            if two_points_distance(poly_a[i % poly_len], poly_a[(i + 1) % poly_len]) > 1e-10:
                # comparing the distance from the vertices of polygon to the
                # particles' position, respectively.
                d_v1_a = two_points_distance(poly_a[i % poly_len], position_a)
                d_v1_b = two_points_distance(poly_a[i % poly_len], position_b)
                d_v2_a = two_points_distance(
                    poly_a[(i + 1) % poly_len], position_a)
                d_v2_b = two_points_distance(
                    poly_a[(i + 1) % poly_len], position_b)
                # excluding the edge that is orthogonal to the segment between
                # two particles but not intersection segment of two voronoi
                # cell
                if abs(d_v1_a - d_v1_b) < 1e-10 and abs(d_v2_a - d_v2_b) < 1e-10:
                    return True
    return False


def planned_velocity_verification(theta_p, p_i, p_j):
    """
    in order to avoid the influence by setting the absolute velocity to zero,
    the absolute velocity should be resumed by judging whether the planned
    velocity will increase or decrease the distance of the particle pair
    related to the minimum absolute velocity. If the planned velocity will not
    decrease the distance of the related particle pair generating the planned
    absolute velocity, then the particle can move with the absolute velocity
    under the tolerance.
    :param: v_i--the the planned absolute velocity
            theta_i--the moving direction of the planned velocity
            p_i, p_j: the particle pair generating the absolute velicty
    :rtype: boolean variable
            True--the planned absolute velocity will lead to the decrease of
                the distance of the related particle pair, and will be set
                to zero.
            False--the absolute velocity is preserved as the planned absolute
                velocity.
    """
    planned_velocity = np.array([math.cos(theta_p), math.sin(theta_p)])
    normed_vector_particle_pair = (
        np.array(p_j) - np.array(p_i)) / two_points_distance(p_i, p_j)
    if two_points_distance(p_i, p_j) < rep_margin and 0 < np.dot(planned_velocity, normed_vector_particle_pair) <= 1:
        return True
    elif two_points_distance(p_i, p_j) > att_margin and -1 <= np.dot(planned_velocity, normed_vector_particle_pair) < 0:
        return True
    else:
        return False


def first_layer_neighbor_without_graph(positions):
    """
    According to the current positions, the first layer neighbor set of particles
    is obtained.
    :param: positions: N*2 array representing N particles' 2d coordinates
    :rtype: first_layer_neighbor_set: list with size N, each row records the particle i's
            first layer neighbor set.
    """
    # variable for recording intermediate data
    first_layer_neighbor_set = [0] * N
    # recording the info of delaunay cell with designed data structure
    voronoi_cell = [0] * N
    # recording the starting and ending point for each circular sector
    starting_ending_point = [0] * N
    # recording the starting and ending angle for each circular sector
    starting_ending_angle = [0] * N

    # graphic output
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # plt.axis('scaled') # equal axis
    # i = 0
    # for x,y in positions:
    # plt.plot(x,y, 'ob',markersize=2) # plotting particles
    # plt.text(x+0.005 ,y+0.005 , str(i)) # plotting particles indices
    # i += 1

    # obtaining the neighbors in the sensing range and intersecting points of
    # two sensing range circles
    for i in xrange(N):
        neighbor_set_list = []
        starting_ending_point_list = []
        starting_ending_angle_list = []
        k = 0  # recording the number of neighbor particles
        for j in xrange(N):
            if j != i:
                # distance between i and j
                d = math.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
                if d <= SENSING_RANGE:  # particles i's neighbors
                    k += 1
    #                pos_x = [positions[i][0], positions[j][0]]
    #                pos_y = [positions[i][1], positions[j][1]]
    # plt.plot(pos_x, pos_y, '--b', alpha=0.2)# plotting the links between
    # neighbor particles
                    neighbor_set_list.append(j)
                    # recording all the intersection points of particle i and its nearest neighbors
                    # circle_intersection_point_list.append(cip_a) # the first point of circle i and circle j
                    # circle_intersection_point_list.append(cip_b) # the second point of
                    # circle i and circle j
                    # the data structure of circular segment is
                    # [starting_angle, starting_point, ending_angle,
                    # ending_point]
                    starting_point, ending_point, starting_angle, ending_angle = bisector(
                        positions[i], positions[j], r=SENSING_RANGE)
                    starting_ending_point_list.append(starting_point)
                    starting_ending_point_list.append(ending_point)
                    starting_ending_angle_list.append(
                        starting_angle * 180 / math.pi)
                    starting_ending_angle_list.append(
                        ending_angle * 180 / math.pi)
        # the neighbor particles of particle i
        neighbor_set[i] = neighbor_set_list
        starting_ending_point[i] = starting_ending_point_list
        starting_ending_angle[i] = starting_ending_angle_list

    # according to the set circle_intersection_point[i], ordering the intersection points located from the starting to the ending point of circular segments of particle i,
    # then, constructing an approximated voronoi cell by interpolating the
    # circle with intersection points between starting and ending point of
    # circular segment.
    for i in xrange(N):
        poly_points = []
    # fcolor = np.random.rand(3,1) # setting the color for filling the vn
    # region of particle
        starting_angle = math.atan2(starting_ending_point[i][0][
                                    1] - positions[i][1], starting_ending_point[i][0][0] - positions[i][0])
        ending_angle = math.atan2(starting_ending_point[i][1][
                                  1] - positions[i][1], starting_ending_point[i][1][0] - positions[i][0])
        if starting_angle < 0:
            starting_angle += 2 * math.pi
        if ending_angle < 0:
            ending_angle += 2 * math.pi
        intersection_point_within_arc = []
        for x, y in starting_ending_point[i]:
            current_angle = math.atan2(
                y - positions[i][1], x - positions[i][0])
            if current_angle < 0:
                current_angle += 2 * math.pi
            if starting_angle < ending_angle:
                if current_angle >= starting_angle and current_angle <= ending_angle:
                    intersection_point_within_arc.append([x, y])
            else:
                if current_angle >= starting_angle or current_angle <= ending_angle:
                    intersection_point_within_arc.append([x, y])
        intersection_point_within_arc.sort(
            key=lambda c: math.atan2(c[1] - positions[i][1], c[0] - positions[i][0]))
        poly_points = intersection_point_within_arc
        if len(poly_points) == 2:
            # voronoi cell is a circular segment, so the representative points
            # are the two intersection points
            a = LineString(poly_points)
        else:
            # transfer a set of 2d points to a polygon object
            a = Polygon(poly_points)
        tmp_poly = a
        for j in xrange(1, len(neighbor_set[i])):
            starting_angle = math.atan2(starting_ending_point[i][
                                        2 * j][1] - positions[i][1], starting_ending_point[i][2 * j][0] - positions[i][0])
            ending_angle = math.atan2(starting_ending_point[i][
                                      2 * j + 1][1] - positions[i][1], starting_ending_point[i][2 * j + 1][0] - positions[i][0])
            if starting_angle < 0:
                starting_angle += 2 * math.pi
            if ending_angle < 0:
                ending_angle += 2 * math.pi
            intersection_point_within_arc = []
            for x, y in starting_ending_point[i]:
                current_angle = math.atan2(
                    y - positions[i][1], x - positions[i][0])
                if current_angle < 0:
                    current_angle += 2 * math.pi
                if starting_angle < ending_angle:
                    if current_angle >= starting_angle and current_angle <= ending_angle:
                        intersection_point_within_arc.append([x, y])
                else:
                    if current_angle >= starting_angle or current_angle <= ending_angle:
                        intersection_point_within_arc.append([x, y])
            intersection_point_within_arc.sort(
                key=lambda c: math.atan2(c[1] - positions[i][1], c[0] - positions[i][0]))
            poly_points = intersection_point_within_arc
            if len(poly_points) == 2:
                # voronoi cell is a circular segment, so the representative
                # points are the two intersection points
                a = LineString(poly_points)
            else:
                # transfer a set of 2d points to a polygon object
                a = Polygon(poly_points)
            b = tmp_poly.intersection(a)
            tmp_poly = b
#        patch = PolygonPatch(b, fc=fcolor, ec=fcolor, alpha=0.6, zorder=1)
#        ax.add_patch(patch)
        # finally obtained polygons representing the voronoi cell for particles
        # i
        voronoi_cell[i] = tmp_poly

    # calculating the first layer neighbor particles
    for i in xrange(N):
        first_layer_neighbor_list = []
        for j in neighbor_set[i]:
            # the only one particle in its sensing range is the voronoi-like
            # neighbor
            if len(neighbor_set[i]) == 1 or two_points_distance(positions[i], positions[j]) == 1:
                # and two circles have only one intersection point
                first_layer_neighbor_list.append(j)
#                pos_x = [positions[i][0], positions[j][0]]
#                pos_y = [positions[i][1], positions[j][1]]
# plt.plot(pos_x, pos_y, '--b', alpha=0.2) # plotting the links between
# voronoi-like neighbor particles
            # user-defined function to judge the intersection of two polygons
            elif limited_delaunay_neighbor(voronoi_cell[i], positions[i], positions[j]):
                first_layer_neighbor_list.append(j)
#                pos_x = [positions[i][0], positions[j][0]]
#                pos_y = [positions[i][1], positions[j][1]]
# plt.plot(pos_x, pos_y, '--b', alpha=0.2) # plotting the links between
# voronoi-like neighbor particles
        first_layer_neighbor_set[i] = first_layer_neighbor_list
# setting the region for displaying graph
#    x_max = max(positions[:,0])
#    x_min = min(positions[:,0])
#    y_max = max(positions[:,1])
#    y_min = min(positions[:,1])
#    plt.xlim(x_min-1.1*SENSING_RANGE,x_max+1.1*SENSING_RANGE)
#    plt.ylim(y_min-1.1*SENSING_RANGE,y_max+1.1*SENSING_RANGE)
#    plt.savefig(str(N) +'_particles_sensing_range at ' +str(steps)+' steps.png')

    return first_layer_neighbor_set

# =============================== main function ===============================

# ----------------------------initial process----------------------------------
# 1. reading from existing files containing initial positions if have, or
#    generating initial positions to form a connected graph
# 2. initial files are stored in cPickle form, include config_N_particles.pk,
#    theta_N_particles.pk, v_N_particles.pk
# 3. corresponding to the file reading process, the file writing process is
#    given after each run to save positions, velocities and moving directions
#    into files
# -----------------------------------------------------------------------------
# the directory for saving current simulation results
saving_path = './N' + str(N) + '/ua' + str(u_a) + '/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)
    print 'directory ' + saving_path + ' is created.'

filename_config_pk = 'config_' + \
    str(N) + '_particles.pk'  # file recording portions
if os.path.isfile(saving_path + filename_config_pk):
    positions = cPickle.load(open(saving_path + filename_config_pk, 'rb'))
    print 'starting from file:', filename_config_pk
else:
    ini_positions = [[0, 0]]  # the first particle lies in the origin
    i = 0
    while i < N - 1:
        # % randomly choosing a neighboring particle from existing ones
        index = random.randint(0, i)
        # particle position is given by the polar coordinate,
        # [CORE_RANGE+upsilon, SENSING_RANGE-upsilon]
        rand_r = random.uniform(CORE_RANGE + upsilon, SENSING_RANGE - upsilon)
        rand_theta = random.uniform(-math.pi, math.pi)  # [-pi, pi]
        # the newly generated particles keeps connectivity with the
        # particle(index)
        tmp_x = ini_positions[index][0] + rand_r * math.cos(rand_theta)
        tmp_y = ini_positions[index][1] + rand_r * math.sin(rand_theta)

        # judging the collision between old and new particles
        for j in xrange(i + 1):
            distance = math.sqrt(
                (tmp_x - ini_positions[j][0]) ** 2 + (tmp_y - ini_positions[j][1]) ** 2)
            if distance < 2 * CORE_RANGE:
                # flag for whether newly generated particle collide with
                # existing particle
                isCollision = True
                break
            else:
                isCollision = False
        if isCollision == False:
            ini_positions.append([tmp_x, tmp_y])
            i += 1
    positions = np.asarray(ini_positions)
    print 'configuration starting from scratch'

filename_theta_pk = 'theta_' + str(N) + '_particles.pk'  # file recording theta
if os.path.isfile(saving_path + filename_theta_pk):
    theta = cPickle.load(open(saving_path + filename_theta_pk, 'rb'))
    print 'starting from file:', filename_theta_pk
else:
    theta = []
    for m in xrange(N):
        # initial headings in [-pi, pi]
        theta.append(random.uniform(-math.pi, math.pi))
    print 'theta starting from scratch'

filename_v_pk = 'v_' + str(N) + '_particles.pk'  # file recording velocity
if os.path.isfile(saving_path + filename_v_pk):
    v = cPickle.load(open(saving_path + filename_v_pk, 'rb'))
    print 'starting from file:', filename_v_pk
else:
    v = [V_MAX] * N  # Initial absolute velocities are all V_MAX
    print 'v starting from scratch'


# reading the current running steps in the file running_steps.txt, whose
# content is the total steps the code running
# file recording the total running steps. For the first run, 0 is written
# into the file
filename_running_steps = 'running_steps.txt'
if not os.path.isfile(saving_path + filename_running_steps):
    print filename_running_steps + ' is created for the first time.'
    # writing 0 into file
    f_rs = open(saving_path + filename_running_steps, 'w')
    f_rs .write(str(0))
    f_rs.close()

# reading total running steps from file
f_rs = open(saving_path + filename_running_steps, 'r')
running_steps = int(f_rs.read())
f_rs.close()

# defining file names for storing the data of statistical variables
filename_ord_para = 'order_parameter_' + \
    str(NSTEPS) + '_steps_' + str(N) + \
    '_particles.txt'  # file recording order parameter
# file recording absolute velocities
filename_v_list = 'v_list_' + str(N) + '_particles.txt'
# file recording moving headings
filename_theta_list = 'theta_list_' + str(N) + '_particles.txt'
# file recording the number of first layer neighbors
filename_fln_list = 'fln_list_' + str(N) + '_particles.txt'

# ----------------------- main loop for NSTEPS---------------------------------
# evolution steps
# 1. planning the absolute moving velocities according to the current
#    configuration (relative distances of particles)
# 2. calculating the average heading according to current neighbors'
#    information, including current moving directions and absolute velocities
# 3. updating the positions of particles with planed headings and absolute
#    velocities

order_para = []  # defining variable for calculating order parameter
fln_num = [0] * N  # the number of first layer neighbor of a particle
# config_list = np.zeros((NSTEPS,N)) # recording the absolute velocities
# of all particles
# recording the absolute velocities of all particles for calculating the
# order parameter
v_list = np.zeros((NSTEPS, N))
# recording headings of all particles for calculating the order parameter
theta_list = np.zeros((NSTEPS, N))
for steps in xrange(NSTEPS):
    # obtaining the first layer neighbor set for all particles according to their current positions
    # first_layer_neighbor_set = first_layer_neighbor_with_graph(positions) #
    # saving figure of sensing range at each time step
    first_layer_neighbor_set = first_layer_neighbor_without_graph(
        positions)  # without saving figure of sensing range
    tmp_v = [0.0] * N
    tmp_theta = [0.0] * N
    # for using the '+' operation directly
    positions_offset = np.array([[0.0, 0.0]] * N)
    # forces generated by the alignment rule
    alig_force = np.array([[0.0, 0.0]] * N)
    # forces generated by neighbor particles in the repulsive region
    rep_force = np.array([[0.0, 0.0]] * N)
    # forces generated by neighbor particles in the attractive region
    att_force = np.array([[0.0, 0.0]] * N)
    # resultant forces 'initializing forces at each steps'
    resultant_force = np.array([[0.0, 0.0]] * N)
    for i in xrange(N):  # looping for all particles
        # initial max and min distance between two particles
        robust_d = float('inf')
        index_d = i  # for planning the absolute velocity
        k = 0  # recording the number of neighbor particles
        # looping for all the first layer neighbors of particle i
        for j in first_layer_neighbor_set[i]:
            # distance between i and j
            d = two_points_distance(positions[i], positions[j])
            if d < 2 * CORE_RANGE:  # for debugging
                print 'two particles ' + str(i) + ' and ' + str(j) + ' collide at ' + str(steps) + ' step!'
                print 'distance:', d
            # calculating the min and max distance between particle i and its
            # neighbors
            robust_d_current = min(
                (d - 2 * CORE_RANGE) / 2.0, (SENSING_RANGE - d) / 2.0)
            if robust_d_current < robust_d:
                robust_d = robust_d_current
                index_d = j
            k += 1
            # alignment force without particle i itself
            alig_force[
                i] += [v[j] * math.cos(theta[j]), v[j] * math.sin(theta[j])]
            if d < rep_margin:
                # normalized repulsive force
                rep_force[i] += [u_a * (d - rep_margin) ** 2 / (rep_margin - 2 * CORE_RANGE) ** 2 * (positions[i][0] - positions[j][0]) / d,
                                 u_a * (d - rep_margin) ** 2 / (rep_margin - 2 * CORE_RANGE) ** 2 * (positions[i][1] - positions[j][1]) / d]
#                 rep_force += [(rep_margin-d)/(rep_margin-2*CORE_RANGE)*(positions[i][0]-positions[j][0])/d,\
#                               (rep_margin-d)/(rep_margin-2*CORE_RANGE)*(positions[i][1]-positions[j][1])/d]
            if d > att_margin:
                # normalized attractive force
                att_force[i] += [-u_a * (d - att_margin) ** 2 / (SENSING_RANGE - att_margin) ** 2 * (positions[i][0] - positions[j][0]) / d,
                                 -u_a * (d - att_margin) ** 2 / (SENSING_RANGE - att_margin) ** 2 * (positions[i][1] - positions[j][1]) / d]
#                 att_force[i] += [-(d-att_margin)/(SENSING_RANGE-att_margin)*(positions[i][0]-positions[j][0])/d, \
#                                  -(d-att_margin)/(SENSING_RANGE-att_margin)*(positions[i][1]-positions[j][1])/d]
        # alignment force with particle i itself
        alig_force[i] += [v[i] * math.cos(theta[i]), v[i] * math.sin(theta[i])]
        resultant_force[i] = u_b * alig_force[i] + rep_force[i] + att_force[i]
        tmp_theta[i] = math.atan2(
            resultant_force[i][1], resultant_force[i][0])  # theta = atan2(y,x)
        tmp_v[i] = min(V_MAX, robust_d)
        # verification the absolute velocity to avoid float number inaccuracy
        if tmp_v[i] < V_TOL and planned_velocity_verification(tmp_theta[i], positions[i], positions[index_d]):
            tmp_v[i] = 0.0
        # updating particles' positions (forward updating)
        positions_offset[i] = [
            tmp_v[i] * math.cos(tmp_theta[i]), tmp_v[i] * math.sin(tmp_theta[i])]
# norm_positions_offset = math.sqrt(positions_offset[i][0]**2 + positions_offset[i][1]**2)  # the norm of position_offset
#       Here, the norm of positions_offset is not equal to the tmp_v because of the inaccuracy of float points. Then, after updating the particles' positions,
#       the disconnectivity and collision will happen between two particles. Therefore, the correction method of composing absolute velocity is the
#       truncation of float numbers to make the norm of position offsets less than the planned absolute velocity.
# trunc_digi = 1e16 # the digit for truncating components of the velocity vector
# if tmp_v[i] < V_TOL: # doing correction operation until the norm of position updating vector is less than the planned absolute velocity
# tmp_vx = '%.36f' %(tmp_v[i]*math.cos(tmp_theta[i]))
# tmp_vy = '%.36f' %(tmp_v[i]*math.sin(tmp_theta[i]))
#            if positions_offset[i][0] > 0:
#                tmp_vx = math.floor(positions_offset[i][0] * trunc_digi) / trunc_digi
#            else:
#                tmp_vx = math.ceil(positions_offset[i][0] * trunc_digi) / trunc_digi
#            if positions_offset[i][1] > 0:
#                tmp_vy = math.floor(positions_offset[i][1] * trunc_digi) / trunc_digi
#            else:
#                tmp_vy = math.ceil(positions_offset[i][1] * trunc_digi) / trunc_digi
# positions_offset[i] = [tmp_vx, tmp_vy] # composing the corrected vector for updating positions
# norm_positions_offset = math.sqrt(positions_offset[i][0]**2 + positions_offset[i][1]**2) # the norm of position_offset
#            trunc_digi /= 10
#            print 'particle ' +str(i)+ ' at ' +str(steps)+ ' steps correct its moving vector'
#            print 'positions_offset truncated at ' +str(trunc_digi)+ ' digits'
#            print norm_positions_offset
#        print 'final value of norm_positions_offset:',norm_positions_offset

    # updating the state variables according to the tmp variables
    for i in xrange(N):
        v[i] = tmp_v[i]
        theta[i] = tmp_theta[i]
        # Here, the updating of positions will lead to disconnectivity because
        # of the inaccuracy of float numbers.
        positions[i] += positions_offset[i]
    norm_sum_moving_vectors = math.sqrt(sum(v[i] * math.cos(theta[i]) for i in xrange(
        N)) ** 2 + sum(v[i] * math.sin(theta[i]) for i in xrange(N)) ** 2)
    # calculating order parameter
    order_para.append(1 / float(V_MAX * N) * norm_sum_moving_vectors)
    v_list[steps] = v  # recording absolute velocities of current step
    theta_list[steps] = theta  # recording headings of current step

# ----------------- post process for saving  intermediate results -------------
# --------------- connectivity and symmetry checking for debugging ------------
    # current running steps plus already running steps
    total_steps = steps + running_steps
    # checking the symmetry and connectivity of MAS at every TIME_INTERVAL step
    if steps % TEST_INTERVAL == 0:
        # judging the symmetry of the first layer neighbor set
        degree_matrix = np.zeros((N, N), dtype=int)
        adjacent_matrix = np.zeros((N, N), dtype=int)
        for i in xrange(len(first_layer_neighbor_set)):
            for j in xrange(len(first_layer_neighbor_set[i])):
                adjacent_matrix[i][first_layer_neighbor_set[i][j]] = 1
                degree_matrix[i][i] += 1
        sym_judging = (adjacent_matrix.transpose() == adjacent_matrix)
        flag_sym = True
        for i in xrange(len(first_layer_neighbor_set)):
            for j in xrange(len(first_layer_neighbor_set)):
                if sym_judging[i][j] == False:
                    print 'asymmetric pair:'
                    print i, j
                    flag_sym = False
        if flag_sym == False:
            print 'adjacent matrix is asymmetric at ' + str(steps) + ' step'

        # judging the symmetry and connectivity of the first layer neighbor set
        laplacian_matrix = degree_matrix - adjacent_matrix
        eigenvalue = np.linalg.eigvals(laplacian_matrix)
        count_zero_eig = 0  # recording the number of 1 in eigenvalue
        for eig in eigenvalue:
            if abs(eig) < 1e-9:
                count_zero_eig += 1
        if count_zero_eig > 1:
            print 'adjacent matrix is not connected at ' + str(steps) + ' step'
            break

    # saving initial state of variables and random seed into files
    if total_steps == 0:
        # saving positions data into intermediate file
        filename_config_ini = 'config of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        # writing positions
        f_config_ini = open(saving_path + filename_config_ini, 'w')
        for a in positions:
            f_config_ini.write(str(a[0]) + ' ' + str(a[1]) + '\n')
        f_config_ini.close()
#        print 'saving initial configuration'

        # saving theta into intermediate file
        filename_theta_ini = 'theta of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        # writing theta
        f_theta_ini = open(saving_path + filename_theta_ini, 'w')
        for b in theta:
            f_theta_ini.write(str(b) + '\n')
        f_theta_ini.close()
#        print 'saving initial theta'

        # saving theta into intermediate file
        filename_v_ini = 'v of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        f_v_ini = open(saving_path + filename_v_ini, 'w')  # writing v
        for c in v:
            f_v_ini.write(str(c) + '\n')
        f_v_ini.close()
#        print 'saving initial v'

        filename_ord_para_ini = 'order param of ' + str(N) + ' particles at ' + str(
            total_steps) + ' steps.txt'  # saving order parameter into intermediate file
        # writing order parameter with additional mode
        f_op_ini = open(saving_path + filename_ord_para_ini, 'a')
        for op in order_para:
            f_op_ini.write(str(op) + '\n')
        f_op_ini.close()
#        print 'saving initial order parameter'
        print 'saving initial states'

        filename_random_seed = 'rand_seed.txt'
        # writing random seed
        f_rand_seed = open(saving_path + filename_random_seed, 'w')
        f_rand_seed.write(str(rs))
        f_rand_seed.close()
        print 'saving random seed'

    # saving the intermediate results for variables
    # 1. configuration of system
    # 2. theta
    # 3. v
    # 4. order parameter
    # plotting the intermediate figures and saving them into files
    # 1. configuration of MAS
    # 2. time evolution of order parameter
    # 3. histogram of order parameter
    # 4. first layer neighbors graph

    # defining file names for intermediate results with Pickle
    filename_config_si_pk = 'config of ' + \
        str(N) + ' particles at ' + str(total_steps) + ' steps.pk'
    filename_theta_si_pk = 'theta of ' + \
        str(N) + ' particles at ' + str(total_steps) + ' steps.pk'
    filename_v_si_pk = 'v of ' + \
        str(N) + ' particles at ' + str(total_steps) + ' steps.pk'
    filename_ord_para_si_pk = 'order param of ' + \
        str(N) + ' particles at ' + str(total_steps) + ' steps.pk'

    # saving files at each SAVE_INTERVAL
    if (total_steps + 1) % SAVE_INTERVAL == 0:
        # saving intermediate results with Pickle
        cPickle.dump(
            positions, open(saving_path + filename_config_si_pk, 'wb'))
        cPickle.dump(theta, open(saving_path + filename_theta_si_pk, 'wb'))
        cPickle.dump(v, open(saving_path + filename_v_si_pk, 'wb'))
        cPickle.dump(
            order_para, open(saving_path + filename_ord_para_si_pk, 'wb'))

        # saving positions data into intermediate file
        filename_config_si = 'config of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        # writing positions
        f_config_si = open(saving_path + filename_config_si, 'w')
        for a in positions:
            f_config_si.write(str(a[0]) + ' ' + str(a[1]) + '\n')
        f_config_si.close()
#        print 'saving intermediate config at ' + str(total_steps) + ' steps'

        # saving theta into intermediate file
        filename_theta_si = 'theta of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        # writing theta
        f_theta_si = open(saving_path + filename_theta_si, 'w')
        for b in theta:
            f_theta_si.write(str(b) + '\n')
        f_theta_si.close()
#        print 'saving intermediate theta at ' + str(total_steps) + ' steps'

        # saving theta into intermediate file
        filename_v_si = 'v of ' + \
            str(N) + ' particles at ' + str(total_steps) + ' steps.txt'
        f_v_si = open(saving_path + filename_v_si, 'w')  # writing v
        for c in v:
            f_v_si.write(str(c) + '\n')
        f_v_si.close()
#        print 'saving intermediate v at ' + str(total_steps) + ' steps'

        # order parameter should be attached to the saved file at the NSTEPS steps
        # step1: copy file 'order param of N particles at (NSTEPS-1) steps' as order param of N particles at (total_steps) steps
        # step2: attach the order_para data to the file
        filename_ord_para_si = 'order param of ' + str(N) + ' particles at ' + str(
            total_steps) + ' steps.txt'  # saving order parameter into intermediate file
        if os.path.exists(saving_path + filename_ord_para):
            shutil.copyfile(
                saving_path + filename_ord_para, saving_path + filename_ord_para_si)
            # writing order parameter with additional mode
            f_op_si = open(saving_path + filename_ord_para_si, 'a')
            for op in order_para:
                f_op_si.write(str(op) + '\n')
            f_op_si.close()
            print 'saving intermediate states at ' + str(total_steps) + ' steps'
        else:
            # writing order parameter with additional mode
            f_op_si = open(saving_path + filename_ord_para_si, 'a')
            for op in order_para:
                f_op_si.write(str(op) + '\n')
            f_op_si.close()
# print 'saving intermediate order parameter at ' + str(total_steps) + '
# steps'
            print 'saving intermediate states at ' + str(total_steps) + ' steps'

        t = xrange(total_steps + 1)
        # figure_1 time evolution of order parameter
        if os.path.isfile(saving_path + filename_ord_para_si):
            f_op_tol = open(saving_path + filename_ord_para_si, 'r')
            order_para_tol = []
            for g in f_op_tol:
                order_para_tol.append(float(g))
            f_op_tol.close()
        else:
            print 'failure to load file %s.' % filename_ord_para
        plt.figure(num=1)
        # plotting particles in the plane = Arrow(0, 0, 5*np.sin(z),
        # 5*np.cos(z))
        plt.plot(t, order_para_tol, 'b-')
        plt.title('Time evolution of order parameter of ' +
                  str(N) + ' at ' + str(total_steps) + ' steps')
        plt.xlabel('t')
        plt.ylabel('order parameter')
        # plotting region is scaled by parameter 1.1 for clearly displaying
        plt.xlim(0, len(order_para_tol) * 1.1)
        plt.ylim(0, 1 * 1.1)
        plt.grid('on')
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps order parameter.png')
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps order parameter.ps')
        plt.close()

        # figure_2 particles configuration
        plt.figure(num=2)
        ax = plt.subplot(111)
        for i in xrange(N):
            x = positions[i][0]  # parameters for plotting particles and arrows
            y = positions[i][1]
            dx = v[i] * math.cos(theta[i])
            dy = v[i] * math.sin(theta[i])
            # plotting particles in the plane = Arrow(0, 0, 5*np.sin(z),
            # 5*np.cos(z))
            plt.plot(x, y, 'ok', markersize=2)
        #    arrow(x, y, 2*dx, 2*dy, fc='red', alpha=0.75, length_includes_head=True, width=0.005, head_width=0.02, \
        # head_length=0.01)# arrows represent the velocity vectors(properly
        # scaled)
            arrows = FancyArrowPatch(posA=(x, y), posB=(x + 35 * dx, y + 35 * dy),
                                     color = 'r',
                                     arrowstyle='-|>',
                                     mutation_scale=100 ** .5,
                                     connectionstyle="arc3")

            ax.add_patch(arrows)
        plt.title('Configuration of ' + str(N) +
                  ' particles at ' + str(total_steps) + ' steps')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('scaled')
        plt.grid('on')
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps configuration.png')
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps configuration.ps')
        plt.close()

        # figure_3 histogram of order parameter
        plt.figure(num=3)
        plt.hist(order_para_tol, bins=50, normed='True',
                 label='order parameter $\phi$', color='blue')
        plt.title('Histogram of order parameter in ' +
                  str(total_steps) + ' steps with u_a = ' + str(u_a))
        plt.xlabel('$\phi$')
        plt.ylabel('$\Psi(\phi)$')
        # plotting region is scaled by parameter 1.2 for clearly displaying
        plt.xlim(-0.1, 1.1)
        plt.grid('on')
        plt.savefig(saving_path + 'Histogram of order parameter with u_a = ' +
                    str(u_a) + ' at ' + str(total_steps) + ' steps.png')
        plt.savefig(saving_path + 'Histogram of order parameter with u_a = ' +
                    str(u_a) + ' at ' + str(total_steps) + ' steps.ps')
        plt.close()

        # figure_4 first layer neighbors of MAS
        # the list recording the indices of neighbor particle
        neighbor_set = [0] * N
        # recording the info of delaunay cell with designed data structure
        first_layer_neighbor_set = [0] * N
        voronoi_cell = [0] * N
    #    circle_intersection_point = [0] * N
        # recording the starting and ending point for each circular sector
        starting_ending_point = [0] * N
        # recording the starting and ending angle for each circular sector
        starting_ending_angle = [0] * N
        # graphic output
        fig = plt.figure(num=4)
        ax = fig.add_subplot(111)
        plt.axis('scaled')  # equal axis
        i = 0
        for x, y in positions:
            plt.plot(x, y, 'ob', markersize=2)  # plotting particles
# plt.text(x+0.005 ,y+0.005 , str(i)) # plotting particles indices
            i += 1

        # obtaining the neighbors in the sensing range and intersecting points
        # of two sensing range circles
        for i in xrange(N):
            neighbor_set_list = []
    #        circle_intersection_point_list = []
            starting_ending_point_list = []
            starting_ending_angle_list = []
    # poly_points = [] # recording the sensing range polygon
    # voronoi_cell_list = [] # recording the element of voronoi cell of particle i
    #        cs_list = []
    # new_voronoi_cell_list = [] # recording the element of new generated voronoi cell by particle i's nearest neighbor for comparing with existing voronoi cell
    #        first_layer_neighbor_list = []
            k = 0  # recording the number of neighbor particles
            for j in xrange(N):
                if j != i:
                    # distance between i and j
                    d = math.sqrt(
                        (positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
                    if d <= SENSING_RANGE:  # particles i's neighbors
                        k += 1
        #                pos_x = [positions[i][0], positions[j][0]]
        #                pos_y = [positions[i][1], positions[j][1]]
        # plt.plot(pos_x, pos_y, '--b', alpha=0.2)# plotting the links between
        # neighbor particles
                        neighbor_set_list.append(j)
                        # recording all the intersection points of particle i and its nearest neighbors
    # circle_intersection_point_list.append(cip_a) # the first point of circle i and circle j
    # circle_intersection_point_list.append(cip_b) # the second point of circle i and circle j
                        # the info of voronoi region formed by particle i and
                        # particle j
                        starting_point, ending_point, starting_angle, ending_angle = bisector(
                            positions[i], positions[j], r=SENSING_RANGE)
                        starting_ending_point_list.append(starting_point)
                        starting_ending_point_list.append(ending_point)
                        starting_ending_angle_list.append(
                            starting_angle * 180 / math.pi)
                        starting_ending_angle_list.append(
                            ending_angle * 180 / math.pi)

    # neighbor_num[i] = k # the number of particle i
            # the neighbor particles of particle i
            neighbor_set[i] = neighbor_set_list
    # circle_intersection_point[i] = circle_intersection_point_list # the
    # intersecting points of particle i's sensing circle with that of its
    # neighbor particles
            starting_ending_point[i] = starting_ending_point_list
            starting_ending_angle[i] = starting_ending_angle_list
    #        cs[i] = cs_list
    #        first_layer_neighbor_set_com[i] = first_layer_neighbor_list_com

        # according to the set circle_intersection_point[i], ordering the intersection points located from the starting to the ending point of circular segments of particle i,
        # then, constructing an approximated voronoi cell by interpolating the circle with intersection points between starting and ending point of circular segment.
        # according to the set circle_intersection_point[i], ordering the intersection points located from the starting to the ending point of circular segments of particle i,
        # then, constructing an approximated voronoi cell by interpolating the
        # circle with intersection points between starting and ending point of
        # circular segment.
        for i in xrange(N):
            poly_points = []
    # fcolor = np.random.rand(3,1) # setting the color for filling the vn
    # region of particle
            starting_angle = math.atan2(starting_ending_point[i][0][
                                        1] - positions[i][1], starting_ending_point[i][0][0] - positions[i][0])
            ending_angle = math.atan2(starting_ending_point[i][1][
                                      1] - positions[i][1], starting_ending_point[i][1][0] - positions[i][0])
            if starting_angle < 0:
                starting_angle += 2 * math.pi
            if ending_angle < 0:
                ending_angle += 2 * math.pi
            intersection_point_within_arc = []
            for x, y in starting_ending_point[i]:
                current_angle = math.atan2(
                    y - positions[i][1], x - positions[i][0])
                if current_angle < 0:
                    current_angle += 2 * math.pi
                if starting_angle < ending_angle:
                    if current_angle >= starting_angle and current_angle <= ending_angle:
                        intersection_point_within_arc.append([x, y])
                else:
                    if current_angle >= starting_angle or current_angle <= ending_angle:
                        intersection_point_within_arc.append([x, y])
            intersection_point_within_arc.sort(
                key=lambda c: math.atan2(c[1] - positions[i][1], c[0] - positions[i][0]))
            poly_points = intersection_point_within_arc
            if len(poly_points) == 2:
                # voronoi cell is a circular segment, so the representative
                # points are the two intersection points
                a = LineString(poly_points)
            else:
                # transfer a set of 2d points to a polygon object
                a = Polygon(poly_points)
            tmp_poly = a
            for j in xrange(1, len(neighbor_set[i])):
                starting_angle = math.atan2(starting_ending_point[i][
                                            2 * j][1] - positions[i][1], starting_ending_point[i][2 * j][0] - positions[i][0])
                ending_angle = math.atan2(starting_ending_point[i][
                                          2 * j + 1][1] - positions[i][1], starting_ending_point[i][2 * j + 1][0] - positions[i][0])
                if starting_angle < 0:
                    starting_angle += 2 * math.pi
                if ending_angle < 0:
                    ending_angle += 2 * math.pi
                intersection_point_within_arc = []
                for x, y in starting_ending_point[i]:
                    current_angle = math.atan2(
                        y - positions[i][1], x - positions[i][0])
                    if current_angle < 0:
                        current_angle += 2 * math.pi
                    if starting_angle < ending_angle:
                        if current_angle >= starting_angle and current_angle <= ending_angle:
                            intersection_point_within_arc.append([x, y])
                    else:
                        if current_angle >= starting_angle or current_angle <= ending_angle:
                            intersection_point_within_arc.append([x, y])
                intersection_point_within_arc.sort(
                    key=lambda c: math.atan2(c[1] - positions[i][1], c[0] - positions[i][0]))
                poly_points = intersection_point_within_arc
                if len(poly_points) == 2:
                    # voronoi cell is a circular segment, so the representative
                    # points are the two intersection points
                    a = LineString(poly_points)
                else:
                    # transfer a set of 2d points to a polygon object
                    a = Polygon(poly_points)
                b = tmp_poly.intersection(a)
                tmp_poly = b
    #        patch = PolygonPatch(b, fc=fcolor, ec=fcolor, alpha=0.6, zorder=1)
    #        ax.add_patch(patch)
            # finally obtained polygons representing the voronoi cell for
            # particles i
            voronoi_cell[i] = tmp_poly

    # according to the info of approximated voronoi cells, plotting the
    # accurate voronoi cells
        for i in xrange(N):
            # setting the color for filling the vn region of particle
            fcolor = np.random.rand(3, 1)
            if voronoi_cell[i].geom_type == 'LineString':
                point = list(voronoi_cell[i].coords)
                m_points = np.array([positions[i], point[0], point[1]])
                poly_points = constrained_sensing_xrange(
                    m_points, SENSING_RANGE / 2.0, RES)
                a = Polygon(poly_points)
                patch = PolygonPatch(
                    a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
                ax.add_patch(patch)
            else:
                # poly_a is a closed polygon, so the last vertex is the same as
                # the first point
                poly_a = np.array(voronoi_cell[i].exterior)
                # the number of vertices in poly_a plus 1, since the last
                # vertex is repeated as the first one for a closed polygon
                poly_len = len(poly_a) - 1
                # the last point of polygons is repeated the first one
                for j in xrange(poly_len):
                    v1 = poly_a[j % poly_len]
                    v2 = poly_a[(j + 1) % poly_len]
                    plot_triangle = False
                    for k in xrange(len(starting_ending_point[i]) / 2):
                        cip_1 = starting_ending_point[i][2 * k]
                        cip_2 = starting_ending_point[i][2 * k + 1]
                        points_of_triangle = [positions[i], v1, v2]
                        if abs(two_points_slope(v1, v2) - two_points_slope(cip_1, cip_2)) < 1e-5:
                            plot_triangle = True
                            break
                    if plot_triangle:
                        fill_triangle(points_of_triangle, ax, fcolor)
                    else:
                        starting_angle = math.atan2(
                            v1[1] - positions[i][1], v1[0] - positions[i][0])
                        ending_angle = math.atan2(
                            v2[1] - positions[i][1], v2[0] - positions[i][0])
                        if starting_angle < 0:
                            starting_angle += 2 * math.pi
                        if ending_angle < 0:
                            ending_angle += 2 * math.pi
                        fill_circular_sector(
                            starting_angle, ending_angle, positions[i], SENSING_RANGE / 2.0, ax, fcolor)
        # calculating the first layer neighbor particles
        for i in xrange(N):
            first_layer_neighbor_list = []
            for j in neighbor_set[i]:
                # the only one particle in its sensing range is the
                # voronoi-like neighbor
                if len(neighbor_set[i]) == 1:
                    first_layer_neighbor_list.append(j)
                    pos_x = [positions[i][0], positions[j][0]]
                    pos_y = [positions[i][1], positions[j][1]]
                    # plotting the links between voronoi-like neighbor
                    # particles
                    plt.plot(pos_x, pos_y, '--b', alpha=0.2)
                # user-defined function to judge the intersection of two
                # polygons
                elif limited_delaunay_neighbor(voronoi_cell[i], positions[i], positions[j]):
                    first_layer_neighbor_list.append(j)
                    pos_x = [positions[i][0], positions[j][0]]
                    pos_y = [positions[i][1], positions[j][1]]
                    # plotting the links between voronoi-like neighbor
                    # particles
                    plt.plot(pos_x, pos_y, '--b', alpha=0.2)
            first_layer_neighbor_set[i] = np.asarray(first_layer_neighbor_list)

        # setting the region for displaying graph
        x_max = max(positions[:, 0])
        x_min = min(positions[:, 0])
        y_max = max(positions[:, 1])
        y_min = min(positions[:, 1])
        plt.title(str(N) + ' particles with their constrained sensing range')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(x_min - 1.1 * SENSING_RANGE, x_max + 1.1 * SENSING_RANGE)
        plt.ylim(y_min - 1.1 * SENSING_RANGE, y_max + 1.1 * SENSING_RANGE)
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps sensing range.png')
        plt.savefig(saving_path + str(N) + ' particles at ' +
                    str(total_steps) + ' steps sensing range.ps')
        plt.close()
        print 'saving intermediate figures at ' + str(total_steps) + ' steps'

# ------------------------------file saving----------------------------------
# saving results of each run into files for further simulation and post
# processing
# the content of files to be saved is:
# 1. positions of particles (the last step)
# 2. theta of particles (the last step)
# 3. absolute velocity of particles (the last step)
# 4. order parameter (list for all steps)
# 5. v_list (list for all steps)
# 6. theta_list (list for all steps)
# ---------------------------------------------------------------------------

cPickle.dump(positions, open(saving_path + filename_config_pk, 'wb'))
cPickle.dump(theta, open(saving_path + filename_theta_pk, 'wb'))
cPickle.dump(v, open(saving_path + filename_v_pk, 'wb'))

# f_config = open(saving_path + filename_config, 'w') # writing positions
# for a in positions:
#   f_config.write(str(a[0]) + ' ' + str(a[1]) + '\n')
# f_config.close()
#
# f_theta = open(saving_path + filename_theta, 'w') # writing theta
# for b in theta:
#   f_theta.write(str(b) + '\n')
# f_theta.close()
#
# f_v = open(saving_path + filename_v, 'w') # writing v
# for c in v:
#   f_v.write(str(c) + '\n')
# f_v.close()
#
# writing order parameter with additional mode
f_op = open(saving_path + filename_ord_para, 'a')
for op in order_para:
    f_op.write(str(op) + '\n')
f_op.close()

# writing v_list with additional mode
f_vl = open(saving_path + filename_v_list, 'a')
for vl in v_list:
    for velo in vl:
        f_vl.write(str(velo) + ' ')
    f_vl.write('\n')
f_vl.close()

# writing theta_list with additional mode
f_tl = open(saving_path + filename_theta_list, 'a')
for tl in theta_list:
    for tt in tl:
        f_tl.write(str(tt) + ' ')
    f_tl.write('\n')
f_tl.close()


# writing the new total running steps into file
f_rs = open(saving_path + filename_running_steps, 'w')
f_rs.write(str(running_steps + NSTEPS))
f_rs.close()

# print the ending time
end_time = time.time() - start_time
print 'Ending: ' + time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
print 'Time Elapsing: ' + str(end_time / 60.0) + ' min'
