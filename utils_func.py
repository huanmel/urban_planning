import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point,MultiPolygon, GeometryCollection,MultiPoint,MultiLineString
from shapely import points

from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shapely.plotting
import networkx as nx
import random
import math

def place_obstacles(container_poly, n_obstacles, max_radius=1, max_side=2, 
                   obstacle_type='mixed', allow_overlap=False):
    """
    Place N obstacles (polygons or circles) within a container polygon with size limits.
    
    Parameters:
    - container_poly: Shapely Polygon (area to place obstacles in)
    - n_obstacles: Number of obstacles to place
    - max_radius: Max radius for circles (degrees or meters, ~20m)
    - max_side: Max side length for polygons (degrees or meters, ~20m)
    - obstacle_type: 'polygon', 'circle', or 'mixed' (random mix of both)
    - allow_overlap: If False, obstacles won’t overlap each other
    
    Returns:
    - GeoDataFrame with obstacle geometries
    """
    if not isinstance(container_poly, Polygon):
        raise ValueError("Input must be a Shapely Polygon")

    # Get container bounds
    x_min, y_min, x_max, y_max = container_poly.bounds

    # Estimate feasible number based on area
    avg_area = (max_radius ** 2 * np.pi + max_side ** 2) / 2  # Average of circle and square
    poly_area = container_poly.area
    max_possible = int(poly_area / avg_area)
    if n_obstacles > max_possible:
        print(f"Warning: Requested {n_obstacles} obstacles, but max possible ~{max_possible}")
        n_obstacles = min(n_obstacles, max_possible)

    obstacles = []
    attempts = 0
    max_attempts = 1000

    while len(obstacles) < n_obstacles and attempts < max_attempts:
        # Random center
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        center = Point(x, y)

        # Choose obstacle type
        if obstacle_type == 'polygon':
            is_circle = False
        elif obstacle_type == 'circle':
            is_circle = True
        else:  # 'mixed'
            is_circle = np.random.choice([True, False])

        # Generate obstacle
        if is_circle:
            radius = np.random.uniform(1, max_radius)  # Min ~5m, max ~20m
            obstacle = center.buffer(radius)
        else:
            # Random quadrilateral around center
            side_x = np.random.uniform(1, max_side)
            side_y = np.random.uniform(side_x*0.5, max_side)
            obstacle = Polygon([
                (x - side_x/2, y - side_y/2),
                (x + side_x/2, y - side_y/2),
                (x + side_x/2, y + side_y/2),
                (x - side_x/2, y + side_y/2)
            ])

        # Check containment and overlap
        if container_poly.contains(obstacle):
            if allow_overlap or not any(obstacle.intersects(obs) for obs in obstacles):
                obstacles.append(obstacle)
        
        attempts += 1

    if len(obstacles) < n_obstacles:
        print(f"Could only place {len(obstacles)} of {n_obstacles} obstacles")

    # # Create GeoDataFrame
    # gdf_obstacles = gpd.GeoDataFrame({
    #     "name": [f"Obstacle {i+1}" for i in range(len(obstacles))],
    #     "geometry": obstacles
    # }, crs=container_poly.crs if hasattr(container_poly, 'crs') else "EPSG:4326")
    obstacles = MultiPolygon(obstacles)
    return obstacles

def gdf_append(gdf,appnd_list_objs,name,restriction_tag='no_build'):
    gdf_add = gpd.GeoDataFrame({
    'name': [f'{name}_{i+1}' for i in range(len(appnd_list_objs))],
    'geometry': appnd_list_objs,
    'restriction': [restriction_tag] * len(appnd_list_objs)
    }, crs=gdf.crs)

    # Append the new GeoDataFrame to gdf_local
    gdf1 = pd.concat([gdf, gdf_add])
    return gdf1

from collections.abc import Iterable

def ensure_list(variable):
    if isinstance(variable, Iterable) and not isinstance(variable, str):
        return variable
    else:
        return [variable]

def plot_zones(build_zones=None,no_build_zones=None,park_zones=None,roads=None,add_zones=None,houses=None,trees=None):
    
    def plot_zone(ax,zone,color,txt=None,fontsize=15):
        if zone.geom_type in ['LineString', 'MultiLineString','LinearRing']:
            shapely.plotting.plot_line(zone,color=color,ax=ax)
        else:
            shapely.plotting.plot_polygon(zone,color=color,ax=ax)
            if txt:
                centroid = zone.representative_point()
                ax.text(centroid.x, centroid.y, txt, fontsize=fontsize, ha='center', va='center', color=color,alpha=1)
          
            
    def plot_zones(ax,zones,color,symbol=None,fontsize=15):
        if zones:
            zones=ensure_list(zones)
            for i,zone in enumerate(zones):
                if zone.geom_type in ['GeometryCollection']:
                    for geom in zone.geoms:
                        plot_zone(ax,geom,color)
                else:
                    if symbol:
                        plot_zone(ax,zone,color,symbol,fontsize)
                    else:
                        plot_zone(ax,zone,color,str(i + 1))
    
                            
    def plot_houses(ax,houses,color):
         # Plot each house and add order number
        if houses:
            for i, house in enumerate(houses):
                shapely.plotting.plot_polygon(house, color=color, ax=ax)
                # Get the centroid of the house to place the order number
                centroid = house.centroid
                ax.text(centroid.x, centroid.y, str(i + 1), fontsize=12, ha='center', va='center', color='red')

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_zones(ax,build_zones,'blue')
    plot_zones(ax,park_zones,'green')
    plot_zones(ax,roads,'black')
    plot_zones(ax,add_zones,'grey')
    plot_zones(ax,no_build_zones,'red')
    plot_houses(ax,houses,'gold')
    plot_zones(ax,trees,'green',"*",fontsize=20)
    # Add legend
    legend_elements = [
        MplPolygon([[0,0]], facecolor='blue', alpha=0.5, label='build area'),
        MplPolygon([[0,0]], facecolor='red', alpha=0.5, label='no build/obstacles'),
        MplPolygon([[0,0]], facecolor='green', alpha=0.5, label='forest/parks'),
        MplPolygon([[0,0]], facecolor='gold', alpha=0.5, label='houses'),
        MplPolygon([[0,0]], facecolor='green', alpha=0.5, label='*** trees'),
        
        plt.Line2D([0], [0], color='black', lw=2, label='roads'),
        # MplPolygon([[0,0]], facecolor='red', alpha=0.7, label='Здания')
    ]
    ax.legend(handles=legend_elements)
    return fig,ax
    
    # build_zones=ensure_list(build_zones)
    # if build_zones:
    #     build_zones=ensure_list(build_zones)
    #     for zone in build_zones:
    #         shapely.plotting.plot_polygon(zone,color='blue')
    # if no_build_zones:
    #     no_build_zones=ensure_list(no_build_zones)
    #     for zone in no_build_zones:
    #         shapely.plotting.plot_polygon(zone,color='red')
        
    # if park_zones:
    #     park_zones=ensure_list(park_zones)
    #     for zone in park_zones:
    #         shapely.plotting.plot_polygon(zone,color='green')
            
    # if roads:
    #     for road in roads:
    #         shapely.plotting.plot_line(road,color='black')
    # if add_zone:
    #     for zone in add_zone:
    #         shapely.plotting.plot_polygon(zone,color='red',alpha=0.5)

from shapely.ops import nearest_points

def nearest_neighbor_sort(zones):
    if not zones:
        return []

    # Calculate the center of coordinates
    # center = Point(sum(house.centroid.x for house in houses) / len(houses),
    #                sum(house.centroid.y for house in houses) / len(houses))
    center=Point(0,0)

    # Find the house with the minimum distance to the map center
    start_zone = min(zones, key=lambda zone: zone.centroid.distance(center))
    sorted_zones = [start_zone]
    zones.remove(start_zone)

    while zones:
        last_zone = sorted_zones[-1]
        # Find the nearest house
        nearest_zone = min(zones, key=lambda zone: last_zone.centroid.distance(zone.centroid))
        sorted_zones.append(nearest_zone)
        zones.remove(nearest_zone)

    return sorted_zones

def adjust_road_around_obstacles(road, obstacles, n_steps=5, threshold=10, move_distance=5, spline_points=50):
    """
    Adjust a road to avoid obstacles by moving segment ends away from them, then smooth with spline.
    
    Parameters:
    - road: Shapely LineString (input road)
    - obstacles_gdf: GeoDataFrame with obstacle polygons
    - step_size: Length of each segment (meters, default 5m)
    - threshold: Min distance to obstacle triggering adjustment (meters, default 10m)
    - move_distance: Distance to move away from obstacle (meters, default 5m)
    - spline_points: Number of points in smoothed spline (default 50)
    
    Returns:
    - Shapely LineString (adjusted and smoothed road)
    """
   

    # Normalize road and obstacles
    # norm_road = normalize(road)
    norm_road=road
    # norm_obstacles = obstacles_gdf.copy()
    # norm_obstacles["geometry"] = obstacles_gdf["geometry"].apply(normalize)
    # obstacles_union = norm_obstacles.geometry.union_all()
    obstacles_union=obstacles

    # Divide road into segments
    road_length = norm_road.length
    step_size  = road_length / n_steps
    points = [norm_road.interpolate(i * step_size) for i in range(n_steps + 1)]
    if points[-1] != norm_road.coords[-1]:
        points[-1] = Point(norm_road.coords[-1])  # Ensure endpoint is exact

    # Adjust points away from obstacles
    adjusted_points = [points[0]]  # Start point stays fixed
    for i in range(1, len(points)):
        current = points[i]
        # Find distance and closest point to obstacles
        dist_to_obstacle = current.distance(obstacles_union)
        if dist_to_obstacle < threshold:
            # closest_point = obstacles_union.interpolate(obstacles_union.project(current))
            closest_points=nearest_points(current,obstacles_union)
            closest_point=closest_points[1]
            # Vector from closest obstacle point to current point
            P_x, P_y = current.x, current.y
            if dist_to_obstacle<0.1:
                P_x=current.x+0.1
                P_y=current.y+0.1
                
            dx = P_x - closest_point.x
            dy = P_y - closest_point.y
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:  # Normalize direction
                dx, dy = dx / length, dy / length
            
                
                  
            # Move away from obstacle
            dist_sc=min(abs(length-threshold),step_size)
            new_x = current.x + dx * dist_sc
            new_y = current.y + dy * dist_sc

            adjusted = Point(new_x, new_y)
            # Ensure within bounds (0-100)
            # adjusted = Point(max(0, min(100, new_x)), max(0, min(100, new_y)))
            adjusted_points.append(adjusted)
        else:
            adjusted_points.append(current)

    # Create rough adjusted path
    rough_road = LineString(adjusted_points)

    # Spline smoothing
    # x = [p.x for p in adjusted_points]
    # y = [p.y for p in adjusted_points]
    # if len(x) < 4:  # Need at least 4 points for cubic spline
    #     print("Not enough points for spline; returning rough path")
    #     return rough_road

    # tck, u = splprep([x, y], s=0, k=3)  # Exact fit, cubic spline
    # smooth_coords = splev(np.linspace(0, 1, spline_points), tck)
    # smooth_road = LineString(list(zip(smooth_coords[0], smooth_coords[1])))

    # Denormalize back to original coordinates
    # final_road = denormalize(smooth_road)
    # final_road = smooth_road
    final_road=rough_road
    return final_road

from scipy.interpolate import splprep, splev
def build_road_around_obstacles(start_point, end_point, obstacles,container_poly=None,buffer=1,move_points=False,smooth=False,spline_points=20):
    """
    Build a road from start to end, avoiding multiple obstacles using a visibility graph.
    
    Parameters:
    - start_point: Shapely Point (start of road)
    - end_point: Shapely Point (end of road)
    - obstacles_gdf: GeoDataFrame with obstacle polygons
    
    Returns:
    - Shapely LineString (road geometry)
    """
    # Combine obstacles into a single geometry for efficiency
    # obstacles = shapely.unary_union(obstacles_gdf.geometry)

    # Collect all nodes: start, end, and obstacle vertices
    nodes = [start_point, end_point]
    for obstacle in obstacles.geoms:
        pnts=points(obstacle.buffer(buffer).minimum_rotated_rectangle.exterior.coords)
        if container_poly:
            if container_poly.contains(pnts).all():
                nodes.extend(pnts)  # Exclude closing point
                
        else:
            nodes.extend(pnts)  # Exclude closing point
        
        # for coord in obstacle.exterior.coords[:-1]:
        #     nodes.extend(Point(coord))  # Exclude closing point
    
    # Create visibility graph
    G = nx.Graph()
    for i, n1 in enumerate(nodes):
        G.add_node(i, pos=(n1.x, n1.y))
    
    # Add edges if they don’t intersect obstacles
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes[i+1:], start=i+1):
            if n1 != n2:  # Skip self-loops
                edge = LineString([n1, n2])
                # Check if edge intersects obstacle interior (allow boundary touching)
                if not edge.intersects(obstacles) or edge.touches(obstacles):
                    dist = n1.distance(n2)
                    G.add_edge(i, j, weight=dist, geometry=edge)

    # Find shortest path
    try:
        start_idx = 0  # Start point is first node
        end_idx = 1    # End point is second node
        path = nx.shortest_path(G, source=start_idx, target=end_idx, weight="weight")
        
        # Construct road from path
        road_coords = [Point(G.nodes[i]["pos"]) for i in path]
        rough_road = LineString(road_coords)
        print(f"road constructed in {len(path)} parts")
        if not smooth:
            return rough_road

        # Spline smoothing
        x = [p.x for p in road_coords]
        y = [p.y for p in road_coords]
        
        # Check for sufficient points
        if len(x) < 4:  # Need at least 4 points for cubic spline
            print("Not enough points for spline; returning rough path")
            return rough_road

        # Fit B-spline
        tck, u = splprep([x, y], s=0, k=3)  # s=0 for exact fit, k=3 for cubic
        smooth_coords = splev(np.linspace(0, 1, spline_points), tck)
        smooth_road = LineString(list(zip(smooth_coords[0], smooth_coords[1])))

        # Clip to container if needed (optional)
        return smooth_road
    except nx.NetworkXNoPath:
        print("No path found around obstacles")
        return LineString([start_point, end_point])  # Fallback to direct (may intersect)

def build_road_around_one_obstacle(start_point, end_point, obstacle):
    """
    Build a road from start to end, detouring around an obstacle polygon.
    
    Parameters:
    - start_point: Shapely Point (start of road)
    - end_point: Shapely Point (end of road)
    - obstacle: Shapely Polygon (obstacle to avoid)
    
    Returns:
    - Shapely LineString (road geometry)
    """
    # Direct road
    direct_road = LineString([start_point, end_point])

    # Check if direct road intersects obstacle
    if not direct_road.intersects(obstacle):
        return direct_road  # No detour needed

    # Find intersection points with obstacle boundary
    intersection = direct_road.intersection(obstacle.boundary)
    
    if intersection.is_empty:
        # Shouldn’t happen if intersects=True, but as a safeguard
        return direct_road

    # Handle intersection types
    if intersection.geom_type == 'Point':
        entry_point = intersection
        exit_point = intersection  # Single point case (tangent)
    elif intersection.geom_type == 'MultiPoint':
        points = list(intersection.geoms)
        if len(points) < 2:
            return direct_road  # Not enough points to detour
        # Sort by distance from start_point
        points = sorted(points, key=lambda p: start_point.distance(p))
        entry_point = points[0]
        exit_point = points[-1]
    else:
        print("Unexpected intersection type; returning direct road")
        return direct_road

    # Get obstacle boundary coordinates
    boundary_coords = list(obstacle.exterior.coords[:-1])  # Exclude closing point

    # Find indices of entry and exit points on boundary (approximate nearest)
    entry_idx = min(range(len(boundary_coords)), 
                   key=lambda i: Point(boundary_coords[i]).distance(entry_point))
    exit_idx = min(range(len(boundary_coords)), 
                  key=lambda i: Point(boundary_coords[i]).distance(exit_point))

    # Determine shortest path along boundary (clockwise or counterclockwise)
    if entry_idx < exit_idx:
        clockwise = boundary_coords[entry_idx:exit_idx + 1]
        counterclockwise = boundary_coords[exit_idx:] + boundary_coords[:entry_idx + 1]
    else:
        clockwise = boundary_coords[entry_idx:] + boundary_coords[:exit_idx + 1]
        counterclockwise = boundary_coords[exit_idx:entry_idx + 1]

    detour = clockwise if len(clockwise) < len(counterclockwise) else counterclockwise

    # Construct full road: start -> entry -> detour -> exit -> end
    road_coords = [start_point] + detour + [end_point]
    detour_road = LineString(road_coords)
    # detour_road = detour_road.buffer(0.00005).simplify(0.00001).exterior
    return detour_road


def place_objects(polygon,restricted_area, n_centers, obj_length, obj_width, buffer_distance=3,randomly=False):
    """
    Place N house centers in a polygon, each with a zone for M houses, avoiding restrictions.
    
    Parameters:
    - polygon: Shapely Polygon (zone to place houses in)
    - n_centers: Number of house centers (N)

    - house_area: Area per house (square degrees, e.g., 0.00001 for ~10m² at this latitude)
    - restrictions_gdf: GeoDataFrame with restriction polygons (e.g., parks, forests)
    -restricted_area - union
    - buffer_distance: Minimum distance from center to zone edge (degrees, ~10m)
    
    Returns:
    - List of Shapely Points (house centers)
    """
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon")
    house_area = obj_length*obj_width
    # Calculate required area per center
    total_house_area =  house_area  # Total area needed for M houses
    # Assume a circular zone; radius = sqrt(area / π) + buffer
    local_radius=np.sqrt(total_house_area / np.pi)
    required_radius = local_radius + buffer_distance
    zone_area = np.pi * required_radius ** 2

    # Check feasibility
    poly_area = polygon.area
    if n_centers * zone_area > poly_area:
        max_centers = math.ceil(poly_area / zone_area)
        print(f"Warning: Requested {n_centers} centers, but max possible ~{max_centers}")
        # n_centers = min(n_centers, max_centers)

    # Get bounds
    x_min, y_min, x_max, y_max = polygon.bounds

    # # Combine restrictions into a single geometry if provided
    # if restrictions_gdf is not None and not restrictions_gdf.empty:
    #     restricted_area = restrictions_gdf.geometry.unary_union
    # else:
    #     restricted_area = None

    # Place centers
    centers = []

    if randomly:
        # centers_grid=place_objs_in_polygon(polygon, restricted_area, circle_radius=house_rad, min_spacing=buffer_distance,use_rect=True)
        _,centers_grid=place_rand_objs_in_poly(n_centers*10,polygon,restricted_area,local_radius,buffer_distance)
    else:
        centers_grid=place_rectangles_along_longest_side(polygon,restricted_area, obj_length, obj_width,gap=buffer_distance)
    
    
    if centers_grid:
        print(f"number of generated centers on grid: {len(centers_grid)}")
        if len(centers_grid)>n_centers:
            # items_to_select=random.sample(centers_grid,n_centers)
            # centers = [item for item in centers_grid if item in items_to_select]
            print('select that are far from restrictions')
            dist_list=[]
            for center in centers_grid:
                dist_to_obstacle = center.distance(restricted_area)
                dist_list.append(dist_to_obstacle)
            
            dist_list = np.array(dist_list)
            sort_index = np.argsort(-dist_list)
            
            centers = [centers_grid[idx] for idx in sort_index[0:n_centers]]
            # centers = centers_grid[int(sort_index[0:n_centers])]
        else:
            centers = centers_grid
    else:
        centers=[]

    if len(centers) < n_centers:
        print(f"Could only place {len(centers)} of {n_centers} centers due to space/restrictions")

    return centers

def place_rand_objs_in_poly(n_centers,polygon,restricted_area,required_radius,gap):
    attempts = 0
    max_attempts = 1000
    x_min, y_min, x_max, y_max = polygon.bounds
    
    centers=[]
    objects=[]
    objects_uni=None
    while len(centers) < n_centers and attempts < max_attempts:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        candidate = Point(x, y)
        
        # Create a circular zone around the candidate
        zone = candidate.buffer(required_radius)
        
        # Check conditions:
        # 1. Zone fits in polygon
        # 2. Zone doesn’t intersect restrictions
        # 3. Zone doesn’t overlap other center zones
        dist_to_obstacle = zone.distance(restricted_area)
        if objects_uni:
            dist_to_objs = zone.distance(objects_uni)
        
        if (polygon.contains(zone) and
            (restricted_area is None or not zone.intersects(restricted_area)) and
            (dist_to_obstacle>gap) and (objects_uni is None or dist_to_objs>gap)):
            # all(zone.distance(Point(c.x, c.y)) >= (required_radius * 2+gap) for c in centers)):
            centers.append(candidate)
            objects.append(zone)
            objects_uni=unary_union(objects)
        
        attempts += 1
        
    return centers, objects

def place_objs_in_polygon(container_poly, restrictions, circle_radius=2, min_spacing=4,use_rect=False):
    """
    Place circles in a polygon, avoiding restrictions, maximizing count with no overlap.
    
    Parameters:
    - container_poly: Shapely Polygon (area to place circles in)
    - restrictions: restriction polygons
    - circle_radius: Radius of each circle (degrees, ~10m)
    - min_spacing: Minimum distance between circle centers (degrees, ~20m, ≥ 2 * radius)
    
    Returns:
    - circle center Points
    """
    if not isinstance(container_poly, Polygon):
        raise ValueError("Input must be a Shapely Polygon")
    
    if not use_rect:
        if min_spacing < 2 * circle_radius:
            raise ValueError("min_spacing must be at least 2 * circle_radius to avoid overlap")

    # Get bounds of container polygon
    x_min, y_min, x_max, y_max = container_poly.bounds
    
    # Generate a dense grid of potential centers
    grid_step = circle_radius /2  # Fine resolution for grid
    x_coords = np.arange(x_min, x_max, grid_step)
    y_coords = np.arange(y_min, y_max, grid_step)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

    # Filter points: must be in container and not in restrictions
    valid_points = []
    for point in grid_points:
        circle = point.buffer(circle_radius)
        if use_rect:
            obj=circle.minimum_rotated_rectangle
        else:
            obj=circle
        
        nearest = nearest_points(obj, restrictions)
        dist = obj.distance(nearest[1])
        
        if (container_poly.contains(obj) and 
            (restrictions is None or not obj.intersects(restrictions))) and (dist>=min_spacing):
            
            valid_points.append(obj)

    if not valid_points:
        print("No valid points found")
        return None

    # Greedy packing: select centers maximizing placement
    centers = []
    while valid_points:
        # Start with the first valid point
        if not centers:
            centers.append(valid_points.pop(0))
        else:
            # Find the point farthest from existing centers
            distances = [min(center.distance(p) for center in centers) for p in valid_points]
            max_dist_idx = np.argmax(distances)
            new_center = valid_points[max_dist_idx]
            
            # Check if it’s far enough from all existing centers
            if distances[max_dist_idx] >= min_spacing:
                centers.append(new_center)
            
            # Remove the selected point
            valid_points.pop(max_dist_idx)
    
       
    return centers

from shapely.ops import nearest_points, unary_union

def distance_meshgrid(container_poly, restriction, grid_size=5):
    """
    Create a mesh grid over a polygon with distances to the nearest restriction point.
    
    Parameters:
    - container_poly: Shapely Polygon (area to grid)
    - restriction: Shapely Polygon or MultiPolygon (obstacles to measure distance from)
    - grid_size: Spacing between grid points (meters, default 5m)
    
    Returns:
    - tuple: (x_coords, y_coords, distances) as NumPy arrays
    """
    if not isinstance(container_poly, Polygon):
        raise ValueError("container_poly must be a Shapely Polygon")
    if not isinstance(restriction, (Polygon, MultiPolygon,GeometryCollection,MultiLineString)):
        raise ValueError("restriction must be a Shapely Polygon or MultiPolygon")

    # Get bounds of the container polygon (in local 0-100 meter system)
    x_min, y_min, x_max, y_max = container_poly.bounds
    
    # Ensure bounds are within 0-100
    # x_min, x_max = max(0, x_min), min(100, x_max)
    # y_min, y_max = max(0, y_min), min(100, y_max)
    
    # Generate mesh grid
    x_coords = np.arange(x_min, x_max + grid_size, grid_size)
    y_coords = np.arange(y_min, y_max + grid_size, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    
    # Filter points inside the container polygon
    valid_points = [p for p in grid_points if container_poly.contains(p)]
    if not valid_points:
        print("No valid points inside container polygon")
        return np.array([]), np.array([]), np.array([])

    # Calculate minimum distance to restriction
    distances = []
    if isinstance(restriction, MultiPolygon):
        restriction_union = unary_union(restriction)
    else:
        restriction_union = restriction
    
    for point in valid_points:
        # Find nearest point on restriction
        nearest = nearest_points(point, restriction_union)
        # nearest[0] is the input point, nearest[1] is the point on restriction
        
        if restriction_union.intersects(point) or restriction_union.contains(point):
            dist = 0
        else:
            dist = point.distance(nearest[1])
        distances.append(dist)
    
    # Reshape distances to match grid
    valid_x = [p.x for p in valid_points]
    valid_y = [p.y for p in valid_points]
    grid_shape = (len(y_coords), len(x_coords))
    distance_grid = np.full(grid_shape, np.nan)  # NaN for points outside polygon
    
    # Map valid points back to grid indices
    x_indices = np.searchsorted(x_coords, valid_x)
    y_indices = np.searchsorted(y_coords, valid_y)
    for xi, yi, dist in zip(x_indices, y_indices, distances):
        distance_grid[yi, xi] = dist
    
    return x_coords, y_coords, distance_grid, valid_points

def plot_mesh_dist(container_poly,restrictions,x_coords, y_coords, distances):
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_container = gpd.GeoDataFrame({"geometry": [container_poly]})
    gdf_restrictions = gpd.GeoDataFrame({"geometry": [restrictions]})

    # Plot container and restrictions
    gdf_container.plot(ax=ax, color="lightblue", alpha=0.5, edgecolor="black")
    gdf_restrictions.plot(ax=ax, color="green", alpha=0.5, edgecolor="black")

    # Plot distance grid as a heatmap
    c = ax.pcolormesh(x_coords, y_coords, distances, cmap="viridis", shading="auto")
    plt.colorbar(c, ax=ax, label="Distance to Nearest Restriction (meters)")

    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    plt.title("Distance Mesh Grid in Local Coordinate System")
    plt.show()

    # Print some sample distances
    # print("Sample distances at grid points:")
    # for i in range(0, len(x_coords), 2):
    #     for j in range(0, len(y_coords), 2):
    #         if not np.isnan(distances[j, i]):
    #             print(f"({x_coords[i]:.0f}, {y_coords[j]:.0f}): {distances[j, i]:.2f} meters")
    
def build_robot_path(start_point, end_point, container_poly, restriction, grid_size=3, spline_points=50, smooth=True):
    """
    Find a multi-segment path between two points, staying far from obstacles using A* on a mesh grid.
    
    Parameters:
    - start_point: Shapely Point (start)
    - end_point: Shapely Point (end)
    - container_poly: Shapely Polygon (container area)
    - restriction: Shapely Polygon or MultiPolygon (obstacles)
    - grid_size: Grid spacing (meters, default 5m)
    - spline_points: Number of points in smoothed spline (default 50)
    - smooth: If True, apply spline smoothing
    
    Returns:
    - Shapely LineString (optimized multi-segment path)
    """
    # Get mesh grid with distances
    x_coords, y_coords, distance_grid, grid_points = distance_meshgrid(container_poly, restriction, grid_size)
    if distance_grid.size == 0:
        print("No valid grid; returning direct path")
        return LineString([start_point, end_point])

    # Build graph
    G = nx.Graph()
    
    # Add grid points as nodes with distance data
    for i, point in enumerate(grid_points):
        G.add_node(i, pos=(point.x, point.y), distance=distance_grid[
            int(np.searchsorted(y_coords, point.y)),
            int(np.searchsorted(x_coords, point.x))
        ])

    # Add start and end points as nodes
    start_idx = len(grid_points)
    end_idx = start_idx + 1
    G.add_node(start_idx, pos=(start_point.x, start_point.y), distance=start_point.distance(restriction))
    G.add_node(end_idx, pos=(end_point.x, end_point.y), distance=end_point.distance(restriction))

    # Add edges between grid points (8-connected)
    grid_shape = distance_grid.shape
    for i, p1 in enumerate(grid_points):
        p1_idx = (int(np.searchsorted(y_coords, p1.y)), int(np.searchsorted(x_coords, p1.x)))
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nxx = p1_idx[0] + dy, p1_idx[1] + dx
            if 0 <= ny < grid_shape[0] and 0 <= nxx < grid_shape[1] and not np.isnan(distance_grid[ny, nxx]):
                p2 = Point(x_coords[nxx], y_coords[ny])
                j = next(k for k, p in enumerate(grid_points) if p == p2)  # Find index
                edge = LineString([p1, p2])
                if container_poly.contains(edge):
                    length = p1.distance(p2)
                    avg_dist = (G.nodes[i]["distance"] + G.nodes[j]["distance"]) / 2
                    cost = length / (avg_dist + 1e-6)  # Inverse distance as cost
                    G.add_edge(i, j, weight=cost, geometry=edge)

    # Connect start and end points to nearby grid points
    for idx, p in [(start_idx, start_point), (end_idx, end_point)]:
        edge_found=False
        for j, grid_p in enumerate(grid_points):
            edge = LineString([p, grid_p])
            if container_poly.contains(edge) and not edge.intersects(restriction):
                length = p.distance(grid_p)
                avg_dist = (G.nodes[idx]["distance"] + G.nodes[j]["distance"]) / 2
                cost = length / (avg_dist + 1e-6)
                G.add_edge(idx, j, weight=cost, geometry=edge)
                edge_found=True
        if not edge_found:
            print(f"non feasible solution - no edge to connect to start/finish point: {idx}: {p}")
            

    # A* pathfinding
    try:
        # nx.draw(G,pos=nx.spring_layout(G))
        path_indices = nx.astar_path(G, start_idx, end_idx, 
                                    heuristic=lambda a, b: Point(G.nodes[a]["pos"]).distance(Point(G.nodes[b]["pos"])))
        path_coords = [Point(G.nodes[i]["pos"]) for i in path_indices]
        rough_path = LineString(path_coords)
    except nx.NetworkXNoPath:
        print("No path found; returning direct path")
        return LineString([start_point, end_point])

    # Return rough path if not smoothing
    if not smooth:
        return rough_path

    # Spline smoothing
    x = [p.x for p in path_coords]
    y = [p.y for p in path_coords]
    if len(x) < 4:
        return rough_path
    tck, u = splprep([x, y], s=0, k=3)
    smooth_coords = splev(np.linspace(0, 1, spline_points), tck)
    smooth_path = LineString(list(zip(smooth_coords[0], smooth_coords[1])))

    # Clip to container
    final_path = smooth_path.intersection(container_poly)
    if final_path.is_empty or final_path.length < rough_path.length * 0.5:  # Avoid excessive clipping
        return rough_path
    return final_path
    
def rectangular_voronoi(polygon, seed_points):
    """
    Create a Voronoi-like diagram with rectangular regions from seed points within a polygon.
    
    Parameters:
    - polygon: Shapely Polygon (boundary to clip to)
    - seed_points: List of [x, y] coordinates for seeds
    
    Returns:
    - GeoDataFrame with rectangular regions
    """
    # Convert seed points to array
    seeds = np.array(seed_points)
    n_seeds = len(seeds)

    # Get bounds of the polygon
    x_min, y_min, x_max, y_max = polygon.bounds

    # Sort seeds by x-coordinate to split horizontally
    sorted_indices = np.argsort(seeds[:, 0])
    sorted_seeds = seeds[sorted_indices]

    # Divide the x-range into roughly equal segments based on seed positions
    x_splits = [x_min] + [(sorted_seeds[i, 0] + sorted_seeds[i+1, 0]) / 2 
                         for i in range(n_seeds-1)] + [x_max]
    
    # Create rectangular regions
    rectangles = []
    for i in range(n_seeds):
        # Define rectangle bounds
        left = x_splits[i]
        right = x_splits[i + 1]
        # Use full y-range for simplicity; could split vertically too
        rect = Polygon([
            (left, y_min),
            (right, y_min),
            (right, y_max),
            (left, y_max),
            (left, y_min)
        ])
        # Clip to the input polygon
        clipped_rect = rect.intersection(polygon)
        if not clipped_rect.is_empty:
            rectangles.append(clipped_rect)

    # Create GeoDataFrame
    # gdf = gpd.GeoDataFrame({
    #     "name": [f"Region {i+1}" for i in range(len(rectangles))],
    #     "geometry": rectangles
    # }, crs="EPSG:4326")

    return rectangles

from scipy.spatial import cKDTree

def manhattan_rectangular_voronoi(polygon, seed_points):
    # Create a grid of points within the polygon bounds
    x_min, y_min, x_max, y_max = polygon.bounds
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Filter points inside the polygon
    inside = [polygon.contains(Point(p)) for p in grid_points]
    grid_points = grid_points[inside]

    # Use cKDTree with Manhattan distance (p=1)
    tree = cKDTree(seed_points)
    distances, indices = tree.query(grid_points, p=1)  # p=1 for Manhattan distance
    
    # Assign each grid point to nearest seed
    regions = [[] for _ in range(len(seed_points))]
    for pt, idx in zip(grid_points, indices):
        regions[idx].append(pt)

    # Create bounding rectangles
    rectangles = []
    for region_pts in regions:
        if region_pts:
            pts_array = np.array(region_pts)
            rect = Polygon([
                (pts_array[:, 0].min(), pts_array[:, 1].min()),
                (pts_array[:, 0].max(), pts_array[:, 1].min()),
                (pts_array[:, 0].max(), pts_array[:, 1].max()),
                (pts_array[:, 0].min(), pts_array[:, 1].max()),
                (pts_array[:, 0].min(), pts_array[:, 1].min())
            ])
            clipped_rect = rect.intersection(polygon)
            if not clipped_rect.is_empty:
                rectangles.append(clipped_rect)

    # gdf = gpd.GeoDataFrame({
    #     "name": [f"Region {i+1}" for i in range(len(rectangles))],
    #     "geometry": rectangles
    # }, crs="EPSG:4326")
    return rectangles

def combine_zones(zones_in):
    # combine zones
    zones=zones_in.copy()
    zones1=[]
    for zone in zones:
        if zone.geom_type=='MultiPolygon':
            for geom in zone.geoms:
                zones1.append(geom)
        else:
            zones1.append(zone)
    zones=zones1
    zones_neighbours=np.zeros((len(zones),len(zones)))
    for i, zone in enumerate(zones):
        for j, other_zone in enumerate(zones):
            # if i==j:
            #     continue
            if zone.touches(other_zone):
                zones_neighbours[i,j]+=1
            #     zones[i] = zone.union(other_zone)
            #     zones[j+i+1] = None
    print(zones_neighbours)
    zones_neighbours_sum=zones_neighbours.sum(axis=0)
    print(zones_neighbours_sum)
    # squeeze zones with one neighour

    for i, zone in enumerate(zones):
        if zones_neighbours_sum[i]<2:
            neighbr_idx=zones_neighbours[i].argmax()
            zone_neighbour=zones[neighbr_idx]
            zone_new= zone.union(zone_neighbour)
            zones[i]=zone_new
            zones[neighbr_idx]=None
    zones = [x for x in zones if x is not None]
    return zones

def find_longest_side_in_polygon(polygon):
    """Find the longest straight line segment within a polygon
    """
    vertices = list(polygon.exterior.coords)
    # if polygon.interiors:
    #     for ring in polygon.interiors:
    #         vertices.extend(ring.coords[:-1])
    
    max_length = 0
    longest_line = None
    for i in range(1,len(vertices)):
        p1=vertices[i-1]
        p2=vertices[i]
        line = LineString([p1, p2])
        length = line.length
        if length > max_length:
            max_length = length
            longest_line = line
    if not longest_line:
        print('no longest line found')
        longest_line=LineString([vertices[0], vertices[1]])
        
    return longest_line

def find_longest_line_in_polygon(polygon):
    """Find the longest straight line segment within a polygon."""
    vertices = list(polygon.exterior.coords[:-1])
    if polygon.interiors:
        for ring in polygon.interiors:
            vertices.extend(ring.coords[:-1])
    
    max_length = 0
    longest_line = None
    for i, p1 in enumerate(vertices):
        for p2 in vertices[i+1:]:
            line = LineString([p1, p2])
            if polygon.contains(line):
                length = line.length
                if length > max_length:
                    max_length = length
                    longest_line = line
    return longest_line if longest_line else LineString([vertices[0], vertices[1]])

def place_rectangles_along_longest_side(polygon,restrictions, rect_length, rect_width ,gap=0):
    """
    Place rectangles along the longest side of a polygon, filling with parallel rows.
    
    Parameters:
    - polygon: Shapely Polygon (area to fill)
    - rect_length: Length of each rectangle (meters along longest side)
    - rect_width: Width of each rectangle (meters perpendicular to longest side)
    - step_size: Distance between rectangle starts along longest side (meters)
    
    Returns:
    -  rectangle geometries
    """
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon")

    # Step 1: Find the longest line
    longest_line = find_longest_line_in_polygon(polygon)
    longest_side=find_longest_side_in_polygon(polygon)
    longest_line = longest_side
    
    line_start = Point(longest_line.coords[0])
    line_end = Point(longest_line.coords[-1])
    line_length = longest_line.length

    # Direction vector of the longest line
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    mag = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / mag, dy / mag  # Normalize

    # Perpendicular direction (rotate 90° counterclockwise)
    perp_dx = -dy
    perp_dy = dx

    # Step 2: Place rectangles along the longest line
    rectangles = []
    base_rectangles = []
    
    # steps_along = int(np.ceil(line_length / step_size))
    steps_along = int(np.ceil(line_length / (rect_length+gap)))
    step_size = rect_length+gap
    for i in range(steps_along):
        x_base = line_start.x + (i * step_size) * dx
        y_base = line_start.y + (i * step_size) * dy
        if i * step_size > line_length:
            break
        
        # Define first row rectangle
        corners = [
            (x_base, y_base),  # Bottom-left
            (x_base + rect_length * dx, y_base + rect_length * dy),  # Bottom-right
            (x_base + rect_length * dx + rect_width * perp_dx, y_base + rect_length * dy + rect_width * perp_dy),  # Top-right
            (x_base + rect_width * perp_dx, y_base + rect_width * perp_dy)  # Top-left
        ]

        rect = Polygon(corners)
        base_rectangles.append(rect)
        # if polygon.contains(rect) and (restrictions is None or not rect.intersects(restrictions)):
        #     rectangles.append(rect)

    # Step 3: Add parallel rows
    # base_rectangles = rectangles.copy()  # First row as base
    nmax=int(longest_side.length/(rect_width+gap))
    # dirs1=list(range(-1.0*nmax,0,0.5))
    dirs1=np.arange(-1.0*nmax,0,1)
    # dirs2=list(range(1,nmax+1))
    # dirs2=list(range(0.0,nmax+1,0.5))
    dirs2=np.arange(0.0,nmax,1)
    dirs=np.concat((dirs1,dirs2))
    # dirs=[-1]
    # for offset in range(1, max_offset):
        # Offset in both perpendicular directions
    offset=1
    # shapely.plotting.plot_polygon(polygon,color='blue')
    
    for direction in dirs:  # Up and down
        new_rectangles = []
        for base_rect in base_rectangles:
            offset_x = direction * offset * (rect_width+gap) * perp_dx
            offset_y = direction * offset * (rect_width+gap) * perp_dy
            new_corners = [(x + offset_x, y + offset_y) for x, y in base_rect.exterior.coords[:-1]]
            new_rect = Polygon(new_corners)
            # shapely.plotting.plot_polygon(new_rect,color='purple')
            
            if polygon.contains(new_rect) and (restrictions is None or not new_rect.intersects(restrictions)):
                new_rectangles.append(new_rect)
        if new_rectangles:
            rectangles.extend(new_rectangles)
        # else:
        #     break  # Stop if no more fit in this direction


    return rectangles

def separate_voronoi(polygon,n_regions=3):
    # Compute Voronoi diagram


    x_min, y_min, x_max, y_max = polygon.bounds
    seed_points = []
    while len(seed_points) < n_regions:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        if polygon.contains(Point(x, y)):
            seed_points.append([x, y])

    seed_points=np.array(seed_points)
    vor=shapely.voronoi_polygons(MultiPoint(seed_points),extend_to=polygon,only_edges=False)
    vor_edges=shapely.voronoi_polygons(MultiPoint(seed_points),extend_to=polygon,only_edges=True)
    zones=[]
    for reg in vor.geoms:
        regu=reg.intersection(polygon)
        zones.append(regu)
    edges=[]
    for edge in vor_edges.geoms:
        regu=edge.intersection(polygon)
        edges.append(regu)
        
        
    return seed_points,zones,vor,edges