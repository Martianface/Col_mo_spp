# -*- coding: utf-8 -*-
"""
Created on Thu May 08 10:19:53 2014
function for obtaining the first layer neighbor set

Updated on Tue Sep 02 09:29:51 2014
1. revise the judgment of first layer neighbor in function
poly_intersection(poly_a,poly_b,position_a,position_b) without
using the argument poly_b.

2. first eliminate the edges approximating the curves in poly_a, then
determine the first layer neighbor according to whether there is an edge
in poly_a perpendicular to the line linking the position_a and position_b.

@author: Chenlong.He
"""

# import libraries
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import LineString
from descartes import PolygonPatch
import cPickle
#import profile  # for performance analysis

# defining constants
PARTICLE_NUM = 100  # the number of particles
SENSING_RANGE = math.sqrt(1)  # the sensing range of particle
CORE_RANGE = 0.01  # the radius of hard core of particle
NSTEPS = 0  # the number of total steps of simulation
upsilon = 1e-3  # tolerance for radius of particle's polar coordinate
RES = 100  # the number of vertices of polygons approximating curves
PI = math.pi
ZERO = 1e-4
ORIGIN = np.array([0, 0])
u_a = 0.01

saving_path = './N' + str(PARTICLE_NUM) + '/ua' + str(u_a) + '/'
#saving_path = './N' + str(PARTICLE_NUM) + '/'

# defining related functions


def is_equal(a, b):
    '''
    judging the equality of floating numbers
    :para a b: two floating number
    :rtype: boolean True or False
    '''
    return abs(a - b) < ZERO


def two_points_distance(p1, p2):
    '''
    calculating the distance between two points
    :para p1, p2: 2d position coordinates of two points
    :rtype: float distance
    '''
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def two_points_slope(p1, p2):
    '''
    calculating the slope of two points
    :para p1, p2: 2d coordinates of two points
    :rtype: float slope
    '''
    if abs(p1[0] - p2[0]) < 1e-15:
        return float('inf')  # degeneracy of infinite slope
    else:
        return (p2[1] - p1[1]) / (p2[0] - p1[0])


def two_parallel_segments_colinearity(p1, p2, p3, p4):
    if not point_on_segment(p1, p3, p4) and \
       not point_on_segment(p2, p3, p4) and \
       not point_on_segment(p3, p1, p2) and \
       not point_on_segment(p4, p1, p2):
        return False
    else:
        return True


def point_on_segment(p1, p2, p3):
    '''
    determining whether one polygon's vertex is
    on the edge of the other polygon
    :para p1, p2, p3: p1 is the point of one polygon
                      p2 and p3 are points of another
                      polygon, two of which form an edge
    :rtype: boolean True or False
    '''

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


def bisector(p1, p2, r=SENSING_RANGE):
    """
    :param: p1,p2: the coordination of two particles, p1 is the center point,
                   p2 is the neighboring point.
            r: the half sensing range of particles (the default value is r/2)
    :rtype: ip_sa, ip_ea: starting and ending angle of two intersection points
                          between the bisector and circle.
            ip_sp, ip_ep: starting and ending point of two intersection points
                          between the bisector and circle.
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
    ip_sp = [(r / 2.0) * math.cos(ip_sa), (r / 2.0) * math.sin(ip_sa)] + p1
    ip_ep = [(r / 2.0) * math.cos(ip_ea), (r / 2.0) * math.sin(ip_ea)] + p1
    return ip_sp, ip_ep, ip_sa, ip_ea


def circle_intersection(c_1, c_2, r):
    """
    algorithms for obtaining intersections of two circles algorithm refers to: 
    http://www.ambrsoft.com/TrigoCalc/Circles2/Circle2.htm
    :param: c_1, c_2, r: centers of two circles
                       the same radius of two circle
    :rtype: lists for coordinates of two intersecting points
    """
    a = c_1[0]
    b = c_1[1]
    c = c_2[0]
    d = c_2[1]
    d_center = math.sqrt((c - a) ** 2 + (d - b) ** 2)
    if d_center > 2 * r:
        intersect_1 = ORIGIN
        intersect_2 = ORIGIN
        print 'two circles have no intersection.'
    else:
        alpha = 0.25 * \
            math.sqrt((d_center + 2 * r) * d_center ** 2 * (-d_center + 2 * r))
        x_1 = (a + c) / 2.0 + 2 * (b - d) / d_center ** 2 * alpha
        x_2 = (a + c) / 2.0 - 2 * (b - d) / d_center ** 2 * alpha
        y_1 = (b + d) / 2.0 - 2 * (a - c) / d_center ** 2 * alpha
        y_2 = (b + d) / 2.0 + 2 * (a - c) / d_center ** 2 * alpha
        intersect_1 = [x_1, y_1]
        intersect_2 = [x_2, y_2]
    return intersect_1, intersect_2


def segment_intersection(p1, p2, p3, p4):
    '''
    judging if two segments are intersected
    :para: four coordinates of intersection points. first two points are the first intersection line,
           and last two is the second line
    :rtype: if two segments are intersected, then return the coordinates of the intersection point,
             otherwise, return False
    '''
# if p1[0] == p3[0] and p1[1] == p3[1] or p1[0] == p4[0] and p1[1] == p4[1]: # judging coincidence points
# return p1
#        return True
#    elif p2[0] == p3[0] and p2[1] == p3[1] or p2[0] == p4[0] and p2[1] == p4[1]:
# return p2
#        return True
#    else:
    A1 = (p1[1] - p2[1])
    B1 = (p2[0] - p1[0])
    C1 = -(p1[0] * p2[1] - p2[0] * p1[1])
    A2 = (p3[1] - p4[1])
    B2 = (p4[0] - p3[0])
    C2 = -(p3[0] * p4[1] - p4[0] * p3[1])

    D = A1 * B2 - B1 * A2
    Dx = C1 * B2 - B1 * C2
    Dy = A1 * C2 - C1 * A2
    if D != 0:
        x = Dx / D  # solving intersection point by Crammer rule
        y = Dy / D
        # judging intersection point in the boundary region of segments
        if (x < max(min(p1[0], p2[0]), min(p3[0], p4[0]))) or (x > min(max(p1[0], p2[0]), max(p3[0], p4[0]))) and \
           (y < max(min(p1[1], p2[1]), min(p3[1], p4[1]))) or (y > min(max(p1[1], p2[1]), max(p3[1], p4[1]))):
            return False  # segments intersecting but out of bound
        else:
            # return [x,y]
            return True
    else:
        return False


def constrained_sensing_range(positions, r, res):
    '''
    plotting the constrained sensing range of voronoi-like region
    :para positions(3*2 matrix): the first line is the coordinates of particle, and the remaining lines are the
            coordinates of two intersection points of sensing range circles between particle and its neighbors
          r: SENSING_RANGE
          res: the resolution of polygon approximating curve
    :rtype: approx_positions: N*2 array for the vertices of constrained sensing range polygon
    '''
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
    for i in range(1, res - 1):
        interpolation_points.append([positions[0][0] + r * math.cos(
            theta_starting + i * del_alpha), positions[0][1] + r * math.sin(theta_starting + i * del_alpha)])
    # the last intersection point of two circles
    interpolation_points.append(ending_point)
    approx_positions = np.asarray(interpolation_points)
    return approx_positions


def poly_intersection(poly_a, poly_b, position_a, position_b):
    '''
    determine whether two polygons have a common edge:
    1. finding respective segments in poly_a and poly_b which are orthogonal to the segment linking
       the pos_a and pos_b as candidates of the common edge of two polygons
    2. if any, to exclude segments which approximate the curve, by calculating the distance between
       two points(p1, p2 for poly_a and p3, p4 for poly_b) on the candidate segment to pos_a and pos_b.
       If p1 to pos_a is equal to p1 to pos_b, and the same for p2 and for the candidate segment of 
       poly_b, then p1p2 and p3p4 are two the common edges. 
    3. Then, comparing the two candidate common edges by collinearity. If two parallel segments have no
       coincident point, then these two segments are not the common edge of two polygons.  
    :para: poly_a, poly_b: N*2 array represents the polygon
           positions_a, position_b: coordinates of particles of polygon_a and polygon_b
    :rtype: boolean True or False 
    '''

    com_seg_a1 = ORIGIN
    com_seg_a2 = ORIGIN
    com_seg_b1 = ORIGIN
    com_seg_b2 = ORIGIN
    # finding the segment in poly_a, which is orthogonal to the segment
    # between pos_a and pos_b
    # len(ploy)-1 is used to exclude the polygon's last point
    for i in range(len(poly_a) - 1):
        p1 = poly_a[i]
        p2 = poly_a[(i + 1) % len(poly_a)]
        # excluding the coincident point incurred by approximating errors
        if abs(two_points_distance(p1, p2) > 1e-10):
            inner_product = (position_a[
                             0] - position_b[0]) * (p1[0] - p2[0]) + (position_a[1] - position_b[1]) * (p1[1] - p2[1])
            # judging whether two segments are orthogonal
            # perpendicular bisector can be taken as the candidate common edge
            if abs(inner_product) < 1e-9:
                d1 = two_points_distance(position_a, p1)
                d2 = two_points_distance(position_a, p2)
                d3 = two_points_distance(position_b, p1)
                d4 = two_points_distance(position_b, p2)
                # in order to exclude segments approximating curve
                if is_equal(d1, d3) and is_equal(d2, d4):
                    com_seg_a1 = p1
                    com_seg_a2 = p2
                    break

    # finding the segment in poly_b, which is orthogonal to the segment
    # between pos_a and pos_b
    # len(ploy)-1 is used to exclude the polygon's last point
    for i in range(len(poly_b) - 1):
        p1 = poly_b[i]
        p2 = poly_b[(i + 1) % len(poly_b)]
        # excluding the coincident point incurred by approximating errors
        if abs(two_points_distance(p1, p2) > 1e-10):
            inner_product = (position_a[
                             0] - position_b[0]) * (p1[0] - p2[0]) + (position_a[1] - position_b[1]) * (p1[1] - p2[1])
            # judging whether two segments are orthogonal
            # perpendicular bisector can be taken as the candidate common edge
            if abs(inner_product) < 1e-9:
                d1 = two_points_distance(position_a, p1)
                d2 = two_points_distance(position_a, p2)
                d3 = two_points_distance(position_b, p1)
                d4 = two_points_distance(position_b, p2)
                # in order to exclude segments approximating curve
                if is_equal(d1, d3) and is_equal(d2, d4):
                    com_seg_b1 = p1
                    com_seg_b2 = p2
                    break

    # according to the relationship between two common edge candidates,
    # judging whether two polygons are intersected
    if not is_equal(two_points_distance(com_seg_a1, com_seg_a2), 0) and not is_equal(two_points_distance(com_seg_b1, com_seg_b2), 0):
        # two segments have at least one coincidence
        if two_parallel_segments_colinearity(com_seg_a1, com_seg_a2, com_seg_b1, com_seg_b2):
            return True
        else:
            return False
    else:
        return False


def find_major_arc(positions, r):
    '''
    according to center of circle, and intersection points of two circles, find the starting and ending point of a 
    major arc corresponding to two intersection points. 
    :param: positions(3*2 matrix): the first row is the coordinates of particle(center coordinates of circle), and
    the remaining two rows are the coordinates of two intersection points of two sensing range circles between 
    particle and its nearest neighbors
           r: SENSING_RANGE
    :rtype: coordinates of starting and ending points
    '''
    # calculating theta1 belonging to [0,2pi]
    theta1 = math.atan2(
        positions[1][1] - positions[0][1], positions[1][0] - positions[0][0])
    if theta1 < 0:
        theta1 += 2 * math.pi

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

    # return starting_theta, starting_point, ending_theta, ending_point # all
    # the infos are respecting to the local coordinates whose origin is at
    # positions[0]
    # all the infos are respecting to the global coordinates, corresponding
    # angle should be calculated after minus the
    return starting_point, ending_point, starting_angle, ending_angle


def fill_circular_sector(starting_angle, ending_angle, center, r, ax, fcolor):
    '''
    fill a circular sector with random color
    :param: starting_angle, ending_angle -- scalar represented by radian, 
            angles are denoted by the local coordinate, whose origin is center
            center -- the center coordinates of the circle corresponded to the circular sector, 2d vector
            r -- radius of the circular sector
    '''

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
    for i in range(res):
        poly_circular_sector.append(
            [center[0] + r * math.cos(starting_angle + i * del_theta), center[1] + r * math.sin(starting_angle + i * del_theta)])
    # fcolor = np.random.rand(3,1) # setting the color for filling the vn
    # region of particle
    a = Polygon(poly_circular_sector)
    patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
    ax.add_patch(patch)


def fill_triangle(positions, ax, fcolor):
    '''
    plotting an triangle according to the info of vertices.
    :param: positions -- 3*2 matrix representing the three vertices of an triangle
            ax -- related to plotting the graph
            fcolor -- the color for filling the triangle
    '''
    a = Polygon(positions)
    patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
    ax.add_patch(patch)
    # coordinates of center, angle is relative to the center of circle.0.


def limited_delaunay_neighbor(vc_a, position_a, position_b):
    '''
    accroding to the slope of edges composing the voronoi cell, the first layer neighbors are obtained.
    the algorithm is based on the local information of particle i's neighbors.
    :param: vc_a -- the voronoi cell of particle i, shapely object
            position_a -- the coordinates of particle i
            position_b -- the coordinates of particle j
    :rtype: True -- the particle j is the first layer neighbor of particle i
            False -- otherwise
    '''

    # the slpoe of the segment between position_a and position_b
    k_ab = two_points_slope(position_a, position_b)
    if vc_a.geom_type == 'LineString':
        point = list(vc_a.coords)
        segment_slope = two_points_slope(point[0], point[1])
        # degeneracy situation: slope is infinite
        if k_ab == float('inf') and abs(segment_slope) < 1e-15:
            # comparing the distance from the vertices of polygon to the
            # particles' position, respectively.
            d_c_v1 = two_points_distance(point[0], position_a)
            d_c_v2 = two_points_distance(point[0], position_b)
            # excluding the edge that is orthogonal to the segment between two
            # particles but not intersection segment of two voronoi cell
            if abs(d_c_v1 - d_c_v2) < 1e-4:
                return True
        else:
            # the tolerance will lead to the misjudge of the first layer
            # neighbors
            if abs(k_ab * segment_slope + 1) < 1e-4:
                return True
    else:
        # poly_a is a closed polygon, so the last vertex is the same as the
        # first point
        poly_a = np.array(vc_a.exterior)
        # the number of vertices in poly_a plus 1, since the last vertex is
        # repeated as the first one for a closed polygon
        poly_len = len(poly_a) - 1
        # the last point of polygong is repeated the first one
        for i in range(poly_len):
            k_edge = two_points_slope(
                poly_a[(i + 1) % poly_len], poly_a[i % poly_len])
            # degeneracy situation: slope is infinite
            if k_ab == float('inf') and abs(k_edge) < 1e-15:
                # comparing the distance from the vertices of polygon to the
                # particles' position, respectively.
                d_c_v1 = two_points_distance(poly_a[i % poly_len], position_a)
                d_c_v2 = two_points_distance(poly_a[i % poly_len], position_b)
                if is_equal(d_c_v1, d_c_v2):
                    return True
            else:
                # the tolerance will lead to the misjudge of the first layer
                # neighbors
                if abs(k_ab * k_edge + 1) < 1e-4:
                    # comparing the distance from the vertices of polygon to
                    # the particles' position, respectively.
                    d_c_v1 = two_points_distance(
                        poly_a[i % poly_len], position_a)
                    d_c_v2 = two_points_distance(
                        poly_a[i % poly_len], position_b)
                    # excluding the edge that is orthogonal to the segment
                    # between two particles but not intersection segment of two
                    # voronoi cell
                    if abs(d_c_v1 - d_c_v2) < 1e-4:
                        return True
        return False


def new_limited_delaunay_neighbor(vc_a, position_a, position_b):
    '''
    2014.11.17 modified func oflimited_delaunay_neighbor() to avoid misjudgment of first layer neighbors
    accroding to the distance between two particles positions and two vertices of voronoi cells, the first
    layer neighbor of particle i is obtained.
    :param: vc_a -- the voronoi cell of particle i, shapely object
            position_a -- the coordinates of particle i
            position_b -- the coordinates of particle j
    :rtype: True -- the particle j is the first layer neighbor of particle i
            False -- otherwise
    '''

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
        # the last point of polygong is repeated the first one
        for i in range(poly_len):
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


# 2014.11.11 new defined func for plotting the vornonoi graph accurately
# the alpha value in the corresponding funcs fill_triangle() and
# fill_circular_sector() are set to 1.0
def new_first_layer_neighbor(positions):
    '''
    According to the current positions, the first layer neighbor set of particles
    is obtained.
    :para: positions: N*2 array representing N particles' 2d coordinates
    :rtype: first_layer_neighbor_set: list with size N, each row records the particle i's 
            first layer neighbor set.
    '''
    neighbor_set = [
        0] * PARTICLE_NUM  # the list recording the indices of neighbor particle
    first_layer_neighbor_set = [0] * PARTICLE_NUM
    # recording the info of delaunay cell with designed data structure
    voronoi_cell = [0] * PARTICLE_NUM
#    circle_intersection_point = [0] * PARTICLE_NUM
    # recording the starting and ending point for each circular sector
    starting_ending_point = [0] * PARTICLE_NUM
    # recording the starting and ending angle for each circular sector
    starting_ending_angle = [0] * PARTICLE_NUM
    # graphic output
    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1)
    plt.axis('scaled')  # equal axis

    # labeling all the particles
    i = 0
    for x, y in positions:
        plt.plot(x, y, 'ok', markersize=1)  # plotting particles
        plt.text(x + 0.005, y + 0.005, str(i))  # plotting partilces indices
        i += 1

    # obtainning the neighbors in the sensing range and intersecting points of
    # two sensing range circles
    for i in range(PARTICLE_NUM):
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
        for j in range(PARTICLE_NUM):
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
                    cip_a, cip_b = circle_intersection(
                        positions[i], positions[j], SENSING_RANGE / 2.0)
                    neighbor_set_list.append(j)
                    # recording all the intersection points of particle i and its nearest neighbors
# circle_intersection_point_list.append(cip_a) # the first point of circle i and circle j
# circle_intersection_point_list.append(cip_b) # the second point of
# circle i and circle j
                    # composing the input argument for obtaining sensing range
                    # polygon
                    center_and_intersections = [positions[i], cip_a, cip_b]
                    # the infos of voronoi region formed by particle i and
                    # particle j
                    # the data structure of circular segment is
                    # [starting_angle, starting_point, ending_angle,
                    # ending_point]
                    starting_point, ending_point, starting_angle, ending_angle = find_major_arc(
                        center_and_intersections, SENSING_RANGE / 2.0)
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
    for i in range(PARTICLE_NUM):
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
        for j in range(1, len(neighbor_set[i])):
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

# according to the info of approximated voronoi cells, plotting the
# accurate voronoi cells
    for i in range(PARTICLE_NUM):
        # setting the color for filling the vn region of particle
        fcolor = np.random.rand(3, 1)
        if voronoi_cell[i].geom_type == 'LineString':
            point = list(voronoi_cell[i].coords)
            m_points = np.array([positions[i], point[0], point[1]])
            poly_points = constrained_sensing_range(
                m_points, SENSING_RANGE / 2.0, RES)
            a = Polygon(poly_points)
            patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
            ax.add_patch(patch)
        else:
            # poly_a is a closed polygon, so the last vertex is the same as the
            # first point
            poly_a = np.array(voronoi_cell[i].exterior)
            # the number of vertices in poly_a plus 1, since the last vertex is
            # repeated as the first one for a closed polygon
            poly_len = len(poly_a) - 1
            # the last point of polygon is repeated the first one
            for j in range(poly_len):
                v1 = poly_a[j % poly_len]
                v2 = poly_a[(j + 1) % poly_len]
                plot_triangle = False
                for k in range(len(starting_ending_point[i]) / 2):
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
    for i in range(PARTICLE_NUM):
        first_layer_neighbor_list = []
        for j in neighbor_set[i]:
            # the only one particle in its sensing range is the voronoi-like
            # neighbor
            if len(neighbor_set[i]) == 1:
                first_layer_neighbor_list.append(j)
                pos_x = [positions[i][0], positions[j][0]]
                pos_y = [positions[i][1], positions[j][1]]
                plt.plot(pos_x, pos_y, color=((0, 0, 1, 0.5)), marker='o', linestyle='dashed', markerfacecolor=(
                    (0, 0, 0, 0.5)), markersize=1)  # plotting the links between voronoi-like neighbor particles
            # user-defined function to judge the intersection of two polygons
            elif new_limited_delaunay_neighbor(voronoi_cell[i], positions[i], positions[j]):
                first_layer_neighbor_list.append(j)
                pos_x = [positions[i][0], positions[j][0]]
                pos_y = [positions[i][1], positions[j][1]]
                plt.plot(pos_x, pos_y, color=((0, 0, 1, 0.5)), marker='o', linestyle='dashed', markerfacecolor=(
                    (0, 0, 0, 0.5)), markersize=1)  # plotting the links between voronoi-like neighbor particles
        first_layer_neighbor_set[i] = np.asarray(first_layer_neighbor_list)

    for i in range(PARTICLE_NUM):
        x = positions[i][0]  # parameters for plotting particles and arrows
        y = positions[i][1]
#        dx = v[i] * math.cos(theta[i])
#        dy = v[i] * math.sin(theta[i])
        # plotting particles in the plane = Arrow(0, 0, 5*np.sin(z),
        # 5*np.cos(z))
        plt.plot(x, y, 'ok', markersize=3)
# arrow(x, y, 2*dx, 2*dy, fc='red', alpha=0.75, length_includes_head=True, width=0.005, head_width=0.02, \
# head_length=0.01)# arrows represent the velocity vectors(properly scaled)
#        arrows = FancyArrowPatch(posA=(x, y), posB=(x+30*dx, y+30*dy),
#                                color = 'r',
#                                arrowstyle='-|>',
#                                mutation_scale=100**.5,
#                                connectionstyle="arc3")
#
#        ax.add_patch(arrows)

    # setting the region for displaying graph
    x_max = max(positions[:, 0])
    x_min = min(positions[:, 0])
    y_max = max(positions[:, 1])
    y_min = min(positions[:, 1])
#    plt.title(str(PARTICLE_NUM) + ' particles with their constrained sensing range')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min - 1.1 * SENSING_RANGE, x_max + 1.1 * SENSING_RANGE)
    plt.ylim(y_min - 1.1 * SENSING_RANGE, y_max + 1.1 * SENSING_RANGE)
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.png')
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.eps')
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.ps')

    # converting first layer neighbor set to adjacent matrix used to judge the
    # symmetry and connectivity of the matrix
    degree_matrix = np.zeros((PARTICLE_NUM, PARTICLE_NUM), dtype=int)
    adjacent_matrix = np.zeros((PARTICLE_NUM, PARTICLE_NUM), dtype=int)
    for i in range(len(first_layer_neighbor_set)):
        for j in range(len(first_layer_neighbor_set[i])):
            adjacent_matrix[i][first_layer_neighbor_set[i][j]] = 1
            degree_matrix[i][i] += 1
    sym_judging = (adjacent_matrix.transpose() == adjacent_matrix)
    flag_sym = True
    for i in range(len(first_layer_neighbor_set)):
        for j in range(len(first_layer_neighbor_set)):
            if sym_judging[i][j] == False:
                print 'asymmetric pair:'
                print i, j
                flag_sym = False
    if flag_sym == False:
        print 'adjacent matrix is asymmetric!'
    else:
        print 'adjacent matrix is symmetric!'

    # judging the connectivity of the particle system according to the
    # eigenvalue of Laplacian matrix
    laplacian_matrix = degree_matrix - adjacent_matrix
    eigenvalue = np.linalg.eigvals(laplacian_matrix)
    count_zero_eig = 0  # recording the number of 1 in eigenvalue
    for eig in eigenvalue:
        if abs(eig) < 1e-9:
            count_zero_eig += 1
    if count_zero_eig > 1:
        print 'particle system is not connected!'
    else:
        print 'particle system is connected!'

    plt.show()


# 2014.11.11 new defined func for plotting the vornonoi graph accurately
# the alpha value in the corresponding funcs fill_triangle() and
# fill_circular_sector() are set to 1.0
def new_first_layer_neighbor_new_func(positions):
    '''
    According to the current positions, the first layer neighbor set of particles
    is obtained.
    :para: positions: N*2 array representing N particles' 2d coordinates
    :rtype: first_layer_neighbor_set: list with size N, each row records the particle i's 
            first layer neighbor set.
    '''
    neighbor_set = [
        0] * PARTICLE_NUM  # the list recording the indices of neighbor particle
    first_layer_neighbor_set = [0] * PARTICLE_NUM
    # recording the info of delaunay cell with designed data structure
    voronoi_cell = [0] * PARTICLE_NUM
#    circle_intersection_point = [0] * PARTICLE_NUM
    # recording the starting and ending point for each circular sector
    starting_ending_point = [0] * PARTICLE_NUM
    # recording the starting and ending angle for each circular sector
    starting_ending_angle = [0] * PARTICLE_NUM
    # graphic output
    fig = plt.figure(num=2)
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1)
    plt.axis('scaled')  # equal axis

    # labeling all the particles
    i = 0
    for x,y in positions:
       plt.plot(x,y, 'ok',markersize=1) # plotting particles
       plt.text(x+0.005 ,y+0.005 , str(i)) # plotting particles indices
       i += 1

    # obtaining the neighbors in the sensing range and intersecting points of
    # two sensing range circles
    for i in range(PARTICLE_NUM):
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
        for j in range(PARTICLE_NUM):
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
                    # the infos of voronoi region formed by particle i and
                    # particle j
                    # the data structure of circular segment is
                    # [starting_angle, starting_point, ending_angle,
                    # ending_point]
                    starting_point, ending_point, starting_angle, ending_angle = bisector(
                        positions[i], positions[j], SENSING_RANGE)
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
    for i in range(PARTICLE_NUM):
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
        for j in range(1, len(neighbor_set[i])):
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

# according to the info of approximated voronoi cells, plotting the
# accurate voronoi cells
    for i in range(PARTICLE_NUM):
        # setting the color for filling the vn region of particle
        fcolor = np.random.rand(3, 1)
        if voronoi_cell[i].geom_type == 'LineString':
            point = list(voronoi_cell[i].coords)
            m_points = np.array([positions[i], point[0], point[1]])
            poly_points = constrained_sensing_range(
                m_points, SENSING_RANGE / 2.0, RES)
            a = Polygon(poly_points)
            patch = PolygonPatch(a, fc=fcolor, ec=fcolor, alpha=1, zorder=1)
            ax.add_patch(patch)
        else:
            # poly_a is a closed polygon, so the last vertex is the same as the
            # first point
            poly_a = np.array(voronoi_cell[i].exterior)
            # the number of vertices in poly_a plus 1, since the last vertex is
            # repeated as the first one for a closed polygon
            poly_len = len(poly_a) - 1
            # the last point of polygon is repeated the first one
            for j in range(poly_len):
                v1 = poly_a[j % poly_len]
                v2 = poly_a[(j + 1) % poly_len]
                plot_triangle = False
                for k in range(len(starting_ending_point[i]) / 2):
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
    for i in range(PARTICLE_NUM):
        first_layer_neighbor_list = []
        for j in neighbor_set[i]:
            # the only one particle in its sensing range is the voronoi-like
            # neighbor
            if len(neighbor_set[i]) == 1:
                first_layer_neighbor_list.append(j)
                pos_x = [positions[i][0], positions[j][0]]
                pos_y = [positions[i][1], positions[j][1]]
                plt.plot(pos_x, pos_y, color=((0, 0, 1, 0.5)), marker='o', linestyle='dashed', markerfacecolor=(
                    (0, 0, 0, 0.5)), markersize=1)  # plotting the linkes between voronoi-like neighbor particles
            # user-defined function to judge the intersection of two polygons
            elif new_limited_delaunay_neighbor(voronoi_cell[i], positions[i], positions[j]):
                first_layer_neighbor_list.append(j)
                pos_x = [positions[i][0], positions[j][0]]
                pos_y = [positions[i][1], positions[j][1]]
                plt.plot(pos_x, pos_y, color=((0, 0, 1, 0.5)), marker='o', linestyle='dashed', markerfacecolor=(
                    (0, 0, 0, 0.5)), markersize=1)  # plotting the links between voronoi-like neighbor particles
        first_layer_neighbor_set[i] = np.asarray(first_layer_neighbor_list)

    for i in range(PARTICLE_NUM):
        x = positions[i][0]  # parameters for plotting particles and arrows
        y = positions[i][1]
#        dx = v[i] * math.cos(theta[i])
#        dy = v[i] * math.sin(theta[i])
        # plotting particles in the plane = Arrow(0, 0, 5*np.sin(z),
        # 5*np.cos(z))
        plt.plot(x, y, 'ok', markersize=3)
# arrow(x, y, 2*dx, 2*dy, fc='red', alpha=0.75, length_includes_head=True, width=0.005, head_width=0.02, \
# head_length=0.01)# arrows represent the velocity vectors(properly scaled)
#        arrows = FancyArrowPatch(posA=(x, y), posB=(x+30*dx, y+30*dy),
#                                color = 'r',
#                                arrowstyle='-|>',
#                                mutation_scale=100**.5,
#                                connectionstyle="arc3")
#
#        ax.add_patch(arrows)

    # setting the region for displaying graph
    x_max = max(positions[:, 0])
    x_min = min(positions[:, 0])
    y_max = max(positions[:, 1])
    y_min = min(positions[:, 1])
#    plt.title(str(PARTICLE_NUM) + ' particles with their constrained sensing range')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min - 1.1 * SENSING_RANGE, x_max + 1.1 * SENSING_RANGE)
    plt.ylim(y_min - 1.1 * SENSING_RANGE, y_max + 1.1 * SENSING_RANGE)
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.png')
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.eps')
    plt.savefig(saving_path + str(PARTICLE_NUM) +
                ' particles sensing range at ' + str(NSTEPS) + ' steps.ps')

    # converting first layer neighbor set to adjacent matrix used to judge the
    # symmetry and connectivity of the matrix
    degree_matrix = np.zeros((PARTICLE_NUM, PARTICLE_NUM), dtype=int)
    adjacent_matrix = np.zeros((PARTICLE_NUM, PARTICLE_NUM), dtype=int)
    for i in range(len(first_layer_neighbor_set)):
        for j in range(len(first_layer_neighbor_set[i])):
            adjacent_matrix[i][first_layer_neighbor_set[i][j]] = 1
            degree_matrix[i][i] += 1
    sym_judging = (adjacent_matrix.transpose() == adjacent_matrix)
    flag_sym = True
    for i in range(len(first_layer_neighbor_set)):
        for j in range(len(first_layer_neighbor_set)):
            if sym_judging[i][j] == False:
                print 'asymmetric pair:'
                print i, j
                flag_sym = False
    if flag_sym == False:
        print 'adjacent matrix is asymmetric!'
    else:
        print 'adjacent matrix is symmetric!'

    # judging the connectivity of the particle system according to the
    # eigenvalue of Laplacian matrix
    laplacian_matrix = degree_matrix - adjacent_matrix
    eigenvalue = np.linalg.eigvals(laplacian_matrix)
    count_zero_eig = 0  # recording the number of 1 in eigenvalue
    for eig in eigenvalue:
        if abs(eig) < 1e-9:
            count_zero_eig += 1
    if count_zero_eig > 1:
        print 'particle system is not connected!'
    else:
        print 'particle system is connected!'

    plt.show()


# reading particles' positions to plot
filename_config_pk = 'config_' + str(PARTICLE_NUM) + '_particles.pk'  # file recording positions
if os.path.isfile(saving_path + filename_config_pk):
    positions = cPickle.load(open(saving_path + filename_config_pk, 'rb'))
    print 'success to load file %s.' % filename_config_pk
    # calling functions to plot graph
#    fln = first_layer_neighbor(positions)
# new_first_layer_neighbor(positions) # with old func
#    profile.run("new_first_layer_neighbor(positions)")
    new_first_layer_neighbor_new_func(positions)  # with new func
else:
    print 'failure to load file %s.' % filename_config_pk
