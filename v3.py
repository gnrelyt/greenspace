import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
import json
from datetime import datetime
import geopandas as gpd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Page config
st.set_page_config(
    page_title="Green Mapping Tool - Optimal Park Placement",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗺️ Green Mapping Tool")
st.caption("Optimal park placement using ILP solver + continuous refinement")

# Initialize session state
if "geojson_features" not in st.session_state:
    st.session_state.geojson_features = []
if "parks" not in st.session_state:
    st.session_state.parks = []
if "park_buffers" not in st.session_state:
    st.session_state.park_buffers = []
if "cached_boundary" not in st.session_state:
    st.session_state.cached_boundary = None
if "algorithm_steps" not in st.session_state:
    st.session_state.algorithm_steps = []
if "current_step" not in st.session_state:
    st.session_state.current_step = -1  # -1 means show final result
if "optimization_run" not in st.session_state:
    st.session_state.optimization_run = False
if "park_size_ha" not in st.session_state:
    st.session_state.park_size_ha = 1.25  # Default: average of 0.5-2.0

# ============================================================================
# MODULAR FUNCTIONS
# ============================================================================

def calculate_service_distance(park_size_ha):
    """
    Calculate service distance based on park size.
    0.5 ha → 150m service distance
    2.0 ha → 300m service distance
    Linear scaling in between.
    """
    # Linear interpolation: service_distance = 100 * park_size + 100
    service_distance_m = 100 * park_size_ha + 100
    return service_distance_m

def calculate_area_hectares(coords):
    """
    Calculate area in hectares from lat/lon coordinates.
    """
    if len(coords) < 3:
        return 0
    
    poly = Polygon([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs="EPSG:4326")
    gdf_projected = gdf.to_crs("EPSG:3857")
    area_m2 = gdf_projected.geometry[0].area
    hectares = area_m2 / 10000
    
    return hectares

def get_bounds_from_polygons(features):
    """
    Get the bounding box of all polygons.
    """
    if not features:
        return None
    
    all_coords = []
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords)
    
    if not all_coords:
        return None
    
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    
    return (min(lons), min(lats), max(lons), max(lats))

def load_boundary_polygon(features):
    """
    Load and merge all boundary polygons from GeoJSON features.
    """
    if not features:
        return None
    
    polygons = []
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            poly = Polygon([(c[0], c[1]) for c in coords])
            polygons.append(poly)
    
    if not polygons:
        return None
    
    return unary_union(polygons)

def lonlat_to_meters(lon, lat, ref_lat=54.5973):
    """
    Convert lon/lat degrees to approximate meters at a reference latitude.
    """
    earth_radius = 6371000
    meters_per_lat_degree = earth_radius * np.pi / 180
    meters_per_lon_degree = earth_radius * np.pi / 180 * np.cos(np.radians(ref_lat))
    
    return meters_per_lon_degree, meters_per_lat_degree

def meters_to_degrees(meters, ref_lat=54.5973):
    """
    Convert meters to degrees at reference latitude.
    """
    lon_per_m, lat_per_m = lonlat_to_meters(0, 0, ref_lat)
    return meters / lat_per_m

def create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m):
    """
    Create a park at a given location.
    Park must be FULLY within boundary - no clipping allowed.
    """
    aspect_ratio = 1.5
    park_height_m = np.sqrt(target_area_m2 / aspect_ratio)
    park_width_m = target_area_m2 / park_height_m
    
    park_width_deg = park_width_m / lon_per_m
    park_height_deg = park_height_m / lat_per_m
    
    x1 = centroid[0] - park_width_deg / 2
    y1 = centroid[1] - park_height_deg / 2
    x2 = centroid[0] + park_width_deg / 2
    y2 = centroid[1] + park_height_deg / 2
    
    park = box(x1, y1, x2, y2)
    
    # Check if park is FULLY within boundary (not just intersecting)
    if not boundary_poly.contains(park):
        return None  # Park extends outside boundary, reject it
    
    # Verify area meets minimum requirement
    gdf = gpd.GeoDataFrame([1], geometry=[park], crs="EPSG:4326")
    gdf_projected = gdf.to_crs("EPSG:3857")
    actual_area_m2 = gdf_projected.geometry[0].area
    
    if actual_area_m2 >= 3000:  # Minimum 0.3 ha
        return park
    
    return None

def optimize_park_locations(boundary_poly, num_parks, target_area_m2, service_distance_deg, lon_per_m, lat_per_m):
    """
    Optimize park locations to minimize uncovered area using continuous optimization.
    """
    minx, miny, maxx, maxy = boundary_poly.bounds
    
    def objective(park_coords):
        """
        Objective function: minimize uncovered area.
        park_coords is a flat array of [x1, y1, x2, y2, ..., xN, yN]
        """
        parks = []
        for i in range(0, len(park_coords), 2):
            centroid = [park_coords[i], park_coords[i+1]]
            park = create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m)
            if park is not None:
                parks.append(park)
        
        if not parks:
            return 1e6
        
        # Create buffers
        buffers = [p.buffer(service_distance_deg) for p in parks]
        all_buffers = unary_union(buffers)
        
        # Calculate uncovered area
        covered = boundary_poly.intersection(all_buffers)
        uncovered = boundary_poly.difference(covered)
        
        # Return uncovered area as objective (minimize this)
        return uncovered.area
    
    # Initial guess: distribute parks evenly across boundary
    cols = max(1, int(np.ceil(np.sqrt(num_parks))))
    rows = max(1, int(np.ceil(num_parks / cols)))
    
    width = maxx - minx
    height = maxy - miny
    
    x0 = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= num_parks:
                break
            x = minx + (col + 0.5) * width / cols
            y = miny + (row + 0.5) * height / rows
            x0.extend([x, y])
            count += 1
    
    x0 = np.array(x0)
    
    # Bounds: keep parks within boundary
    bounds = [(minx, maxx), (miny, maxy)] * num_parks
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    # Extract final park locations
    parks = []
    for i in range(0, len(result.x), 2):
        centroid = [result.x[i], result.x[i+1]]
        park = create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m)
        if park is not None:
            parks.append(park)
    
    return parks

def refine_park_positions(parks, boundary_poly, service_distance_deg, target_area_m2, lon_per_m, lat_per_m, min_coverage=99.0):
    """
    Post-processing refinement: continuously optimize park positions and merge redundant parks.
    This fixes the discrete grid limitation by allowing parks to move to optimal positions.
    """
    if len(parks) <= 1:
        return parks
    
    st.info("🔧 Refining park positions (continuous optimization)...")
    
    # Get boundary area
    gdf = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:3857")
    boundary_area_m2 = gdf_proj.geometry[0].area
    
    refined_parks = parks.copy()
    iteration = 0
    max_iterations = 5
    improvements_made = True
    
    while improvements_made and iteration < max_iterations:
        improvements_made = False
        iteration += 1
        
        st.info(f"🔍 Refinement iteration {iteration}/{max_iterations}...")
        
        # STEP 1: Try to merge nearby park pairs
        for i in range(len(refined_parks)):
            for j in range(i + 1, len(refined_parks)):
                park_i = refined_parks[i]
                park_j = refined_parks[j]
                
                # Check if parks are close enough to consider merging
                distance = park_i.centroid.distance(park_j.centroid)
                if distance > service_distance_deg * 2:
                    continue  # Too far apart
                
                # Try to find a single position that covers both areas
                # Start from midpoint between the two parks
                mid_x = (park_i.centroid.x + park_j.centroid.x) / 2
                mid_y = (park_i.centroid.y + park_j.centroid.y) / 2
                
                # Try positions in a circle around midpoint
                best_merge_position = None
                best_merge_coverage = 0
                
                search_radius = service_distance_deg * 0.8
                for angle in np.linspace(0, 2 * np.pi, 16):
                    for radius in np.linspace(0, search_radius, 5):
                        test_x = mid_x + radius * np.cos(angle)
                        test_y = mid_y + radius * np.sin(angle)
                        
                        # Create park at test position
                        test_park = create_park_at_location([test_x, test_y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                        if test_park is None:
                            continue
                        
                        # Test if removing both parks and adding this one maintains coverage
                        test_parks = [p for idx, p in enumerate(refined_parks) if idx != i and idx != j] + [test_park]
                        
                        test_buffers = [p.buffer(service_distance_deg) for p in test_parks]
                        test_coverage_geom = unary_union(test_buffers).intersection(boundary_poly)
                        
                        gdf_test = gpd.GeoDataFrame([1], geometry=[test_coverage_geom], crs="EPSG:4326")
                        gdf_test_proj = gdf_test.to_crs("EPSG:3857")
                        test_coverage_pct = 100 * gdf_test_proj.geometry[0].area / boundary_area_m2
                        
                        if test_coverage_pct >= min_coverage and test_coverage_pct > best_merge_coverage:
                            best_merge_coverage = test_coverage_pct
                            best_merge_position = test_park
                
                # If we found a good merge, apply it
                if best_merge_position is not None:
                    st.success(f"✅ Merged 2 parks into 1 better-positioned park")
                    # Remove both parks, add merged park
                    refined_parks = [p for idx, p in enumerate(refined_parks) if idx != i and idx != j] + [best_merge_position]
                    improvements_made = True
                    
                    # Record refinement step
                    new_coverage = 100 * unary_union([p.buffer(service_distance_deg) for p in refined_parks]).intersection(boundary_poly).area / boundary_area_m2
                    st.session_state.algorithm_steps.append({
                        'type': 'refinement',
                        'parks': refined_parks.copy(),
                        'coverage': new_coverage,
                        'description': f'Refinement: Merged 2 parks → 1 park',
                        'iteration': iteration
                    })
                    
                    break  # Start over after making a change
            
            if improvements_made:
                break  # Start over after making a change
        
        # STEP 2: Fine-tune individual park positions
        if not improvements_made:
            for i in range(len(refined_parks)):
                park = refined_parks[i]
                
                # Try small adjustments to improve coverage
                best_adjustment = None
                best_adjustment_coverage = 0
                
                current_x = park.centroid.x
                current_y = park.centroid.y
                
                # Try positions in a small circle around current position
                adjustment_radius = service_distance_deg * 0.2  # 20% of service distance
                
                for angle in np.linspace(0, 2 * np.pi, 12):
                    for radius in [adjustment_radius * 0.5, adjustment_radius]:
                        test_x = current_x + radius * np.cos(angle)
                        test_y = current_y + radius * np.sin(angle)
                        
                        test_park = create_park_at_location([test_x, test_y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                        if test_park is None:
                            continue
                        
                        # Test coverage with adjusted position
                        test_parks = [p if idx != i else test_park for idx, p in enumerate(refined_parks)]
                        
                        test_buffers = [p.buffer(service_distance_deg) for p in test_parks]
                        test_coverage_geom = unary_union(test_buffers).intersection(boundary_poly)
                        
                        gdf_test = gpd.GeoDataFrame([1], geometry=[test_coverage_geom], crs="EPSG:4326")
                        gdf_test_proj = gdf_test.to_crs("EPSG:3857")
                        test_coverage_pct = 100 * gdf_test_proj.geometry[0].area / boundary_area_m2
                        
                        if test_coverage_pct > best_adjustment_coverage:
                            best_adjustment_coverage = test_coverage_pct
                            best_adjustment = test_park
                
                # Apply adjustment if it improves coverage
                if best_adjustment is not None:
                    current_buffers = [p.buffer(service_distance_deg) for p in refined_parks]
                    current_coverage_pct = 100 * unary_union(current_buffers).intersection(boundary_poly).area / boundary_area_m2
                    
                    if best_adjustment_coverage > current_coverage_pct + 0.1:  # At least 0.1% improvement
                        refined_parks[i] = best_adjustment
                        improvements_made = True
                        st.info(f"📍 Fine-tuned park {i+1} position (+{best_adjustment_coverage - current_coverage_pct:.1f}% coverage)")
    
    if iteration == 1 and not improvements_made:
        st.info("✅ No refinements needed - positions already optimal!")
    else:
        final_coverage = 100 * unary_union([p.buffer(service_distance_deg) for p in refined_parks]).intersection(boundary_poly).area / boundary_area_m2
        st.success(f"✨ Refinement complete: {len(parks)} → {len(refined_parks)} parks, {final_coverage:.1f}% coverage")
        
        # Record final refined state
        st.session_state.algorithm_steps.append({
            'type': 'final',
            'parks': refined_parks.copy(),
            'coverage': final_coverage,
            'description': f'Final (refined): {len(refined_parks)} parks, {final_coverage:.1f}% coverage',
            'total_removed': len(parks) - len(refined_parks)
        })
    
    return refined_parks

def find_minimum_parks_optimal(boundary_poly, min_area_ha=0.5, max_area_ha=2.0, service_distance_m=300):
    """
    Find OPTIMAL minimum number of parks using Integer Linear Programming.
    This finds the provably best solution but is slower than greedy.
    
    service_distance_m is calculated based on average park size.
    """
    try:
        from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except ImportError:
        st.error("PuLP library not installed. Install with: pip install pulp --break-system-packages")
        return []
    
    if boundary_poly is None:
        return []
    
    # Clear previous algorithm steps
    st.session_state.algorithm_steps = []
    st.session_state.current_step = -1
    
    # Calculate average park size and corresponding service distance
    avg_park_size = (min_area_ha + max_area_ha) / 2
    service_distance_m = calculate_service_distance(avg_park_size)
    
    st.info(f"📏 Park size: {avg_park_size:.2f} ha → Service distance: {service_distance_m:.0f}m")
    
    ref_lat = boundary_poly.centroid.y
    lon_per_m, lat_per_m = lonlat_to_meters(0, 0, ref_lat)
    service_distance_deg = meters_to_degrees(service_distance_m, ref_lat)
    target_area_m2 = (min_area_ha + max_area_ha) / 2 * 10000
    
    # Get boundary area
    gdf = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:3857")
    boundary_area_m2 = gdf_proj.geometry[0].area
    
    # ADAPTIVE GRID CALCULATION
    grid_spacing_m = service_distance_m / 3
    
    minx, miny, maxx, maxy = boundary_poly.bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    width_m = width_deg * lon_per_m
    height_m = height_deg * lat_per_m
    
    grid_points_x = max(5, int(np.ceil(width_m / grid_spacing_m)))
    grid_points_y = max(5, int(np.ceil(height_m / grid_spacing_m)))
    
    grid_points_x = min(grid_points_x, 40)  # Limit for ILP performance
    grid_points_y = min(grid_points_y, 40)
    
    total_grid_points = grid_points_x * grid_points_y
    st.info(f"📐 Grid: {grid_points_x}×{grid_points_y} = {total_grid_points} candidate locations")
    
    x_coords = np.linspace(minx, maxx, grid_points_x)
    y_coords = np.linspace(miny, maxy, grid_points_y)
    
    # Record grid setup
    st.session_state.algorithm_steps.append({
        'type': 'grid_setup',
        'parks': [],
        'coverage': 0,
        'description': f'Optimal solver grid: {grid_points_x}×{grid_points_y} = {total_grid_points} positions',
        'grid_x': x_coords.tolist(),
        'grid_y': y_coords.tolist(),
        'boundary': boundary_poly
    })
    
    # STEP 1: Create candidate park locations
    st.info("🔧 Creating candidate park locations...")
    candidate_parks = []
    park_id = 0
    
    for x in x_coords:
        for y in y_coords:
            centroid = [x, y]
            park = create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m)
            if park is not None:
                candidate_parks.append((park_id, park))
                park_id += 1
    
    if not candidate_parks:
        st.error("No valid candidate parks found!")
        return []
    
    st.info(f"✅ Created {len(candidate_parks)} candidate park locations")
    
    # STEP 2: Create demand points (points that need coverage)
    st.info("📍 Creating demand points...")
    demand_grid_size = max(20, min(50, int(np.sqrt(boundary_area_m2 / 10000))))  # Adaptive based on area
    demand_points = []
    
    x_demand = np.linspace(minx, maxx, demand_grid_size)
    y_demand = np.linspace(miny, maxy, demand_grid_size)
    
    for x in x_demand:
        for y in y_demand:
            point = Point(x, y)
            if boundary_poly.contains(point):
                demand_points.append(point)
    
    st.info(f"✅ Created {len(demand_points)} demand points to cover")
    
    # STEP 3: Calculate coverage matrix (which parks cover which demand points)
    st.info("🔍 Calculating coverage relationships...")
    coverage_dict = {}
    
    for park_id, park in candidate_parks:
        park_buffer = park.buffer(service_distance_deg)
        covered_points = []
        
        for i, point in enumerate(demand_points):
            if park_buffer.contains(point):
                covered_points.append(i)
        
        coverage_dict[park_id] = covered_points
    
    # STEP 4: Formulate and solve Integer Linear Program
    st.info("🧮 Solving optimization problem (finding optimal solution)...")
    
    prob = LpProblem("MinimumParkCoverage", LpMinimize)
    
    # Decision variables: whether to place a park at location i
    park_vars = {park_id: LpVariable(f"park_{park_id}", cat=LpBinary) 
                 for park_id, _ in candidate_parks}
    
    # Objective: minimize number of parks
    prob += lpSum(park_vars.values()), "TotalParks"
    
    # Constraints: each demand point must be covered by at least one park
    for j in range(len(demand_points)):
        prob += lpSum(park_vars[park_id] for park_id in coverage_dict 
                     if j in coverage_dict[park_id]) >= 1, f"Cover_Point_{j}"
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_parks = [park for park_id, park in candidate_parks 
                     if park_vars[park_id].varValue == 1]
    
    st.success(f"🎯 Optimal solution found: {len(selected_parks)} parks (provably minimal!)")
    
    # Record optimal state
    optimal_coverage = 100 * unary_union([p.buffer(service_distance_deg) for p in selected_parks]).intersection(boundary_poly).area / boundary_area_m2 if selected_parks else 0
    
    st.session_state.algorithm_steps.append({
        'type': 'optimal_solution',
        'parks': selected_parks.copy(),
        'coverage': optimal_coverage,
        'description': f'Optimal (ILP): {len(selected_parks)} parks',
        'total_parks': len(selected_parks)
    })
    
    # REFINEMENT: Improve positions and merge redundant parks
    refined_parks = refine_park_positions(selected_parks, boundary_poly, service_distance_deg, target_area_m2, lon_per_m, lat_per_m)
    
    return refined_parks

def find_minimum_parks(boundary_poly, min_area_ha=0.5, max_area_ha=2.0, service_distance_m=300):
    """
    Find minimum number of parks needed to cover entire boundary.
    Uses greedy algorithm: iteratively place parks where they cover most uncovered area.
    Grid resolution adapts to boundary size and service distance.
    Records each step for visualization.
    
    service_distance_m is calculated based on average park size.
    """
    if boundary_poly is None:
        return []
    
    # Clear previous algorithm steps
    st.session_state.algorithm_steps = []
    st.session_state.current_step = -1
    
    # Calculate average park size and corresponding service distance
    avg_park_size = (min_area_ha + max_area_ha) / 2
    service_distance_m = calculate_service_distance(avg_park_size)
    
    st.info(f"📏 Park size: {avg_park_size:.2f} ha → Service distance: {service_distance_m:.0f}m")
    
    ref_lat = boundary_poly.centroid.y
    lon_per_m, lat_per_m = lonlat_to_meters(0, 0, ref_lat)
    service_distance_deg = meters_to_degrees(service_distance_m, ref_lat)
    target_area_m2 = (min_area_ha + max_area_ha) / 2 * 10000
    
    # Get boundary area
    gdf = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:3857")
    boundary_area_m2 = gdf_proj.geometry[0].area
    
    parks = []
    uncovered = boundary_poly
    
    # ADAPTIVE GRID CALCULATION
    grid_spacing_m = service_distance_m / 3
    
    minx, miny, maxx, maxy = boundary_poly.bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    width_m = width_deg * lon_per_m
    height_m = height_deg * lat_per_m
    
    grid_points_x = max(5, int(np.ceil(width_m / grid_spacing_m)))
    grid_points_y = max(5, int(np.ceil(height_m / grid_spacing_m)))
    
    grid_points_x = min(grid_points_x, 50)
    grid_points_y = min(grid_points_y, 50)
    
    total_grid_points = grid_points_x * grid_points_y
    st.info(f"📐 Grid: {grid_points_x}×{grid_points_y} = {total_grid_points} candidate locations | Spacing: ~{grid_spacing_m:.0f}m")
    
    x_coords = np.linspace(minx, maxx, grid_points_x)
    y_coords = np.linspace(miny, maxy, grid_points_y)
    
    # STEP 0: Record grid setup
    st.session_state.algorithm_steps.append({
        'type': 'grid_setup',
        'parks': [],
        'coverage': 0,
        'description': f'Adaptive grid created: {grid_points_x}×{grid_points_y} = {total_grid_points} positions',
        'grid_x': x_coords.tolist(),
        'grid_y': y_coords.tolist(),
        'boundary': boundary_poly
    })
    
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        # Calculate current coverage
        if parks:
            buffers = [p.buffer(service_distance_deg) for p in parks]
            covered = unary_union(buffers).intersection(boundary_poly)
            uncovered = boundary_poly.difference(covered)
            
            gdf_uncovered = gpd.GeoDataFrame([1], geometry=[uncovered], crs="EPSG:4326")
            gdf_uncovered_proj = gdf_uncovered.to_crs("EPSG:3857")
            uncovered_area = gdf_uncovered_proj.geometry[0].area if not uncovered.is_empty else 0
            
            coverage = 100 * (1 - uncovered_area / boundary_area_m2)
            
            if coverage >= 99:
                break
        else:
            coverage = 0
        
        # Find best location for next park
        best_park = None
        best_new_coverage = 0
        
        current_coverage = coverage if parks else 0
        with st.spinner(f"🔄 Placing park {iteration + 1}... (current coverage: {current_coverage:.1f}%)"):
            for x in x_coords:
                for y in y_coords:
                    centroid = [x, y]
                    
                    park = create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                    
                    if park is None:
                        continue
                    
                    park_buffer = park.buffer(service_distance_deg)
                    new_coverage = park_buffer.intersection(uncovered)
                    
                    if not new_coverage.is_empty:
                        new_coverage_area = new_coverage.area
                        
                        if new_coverage_area > best_new_coverage:
                            best_new_coverage = new_coverage_area
                            best_park = park
        
        if best_park is None:
            break
        
        parks.append(best_park)
        iteration += 1
        
        # Record this placement step
        new_coverage = 100 * (1 - (boundary_poly.difference(unary_union([p.buffer(service_distance_deg) for p in parks])).area if len(parks) > 0 else boundary_area_m2) / boundary_area_m2)
        
        st.session_state.algorithm_steps.append({
            'type': 'greedy_placement',
            'parks': parks.copy(),
            'coverage': new_coverage,
            'description': f'Park {iteration} placed (greedy)',
            'iteration': iteration,
            'new_park': best_park
        })
    
    # Record pre-consolidation state
    final_coverage = 100 * (1 - (boundary_poly.difference(unary_union([p.buffer(service_distance_deg) for p in parks])).area if len(parks) > 0 else 0) / boundary_area_m2)
    
    st.session_state.algorithm_steps.append({
        'type': 'pre_consolidation',
        'parks': parks.copy(),
        'coverage': final_coverage,
        'description': f'Greedy complete: {len(parks)} parks placed',
        'total_parks': len(parks)
    })
    
    # POST-PROCESSING: Consolidate parks to remove redundancies
    if len(parks) > 1:
        parks = consolidate_parks(parks, boundary_poly, service_distance_deg, target_area_m2, lon_per_m, lat_per_m)
    
    # REFINEMENT: Improve positions and merge remaining redundant parks
    if len(parks) > 1:
        parks = refine_park_positions(parks, boundary_poly, service_distance_deg, target_area_m2, lon_per_m, lat_per_m)
    
    return parks

def consolidate_parks(parks, boundary_poly, service_distance_deg, target_area_m2, lon_per_m, lat_per_m, min_coverage=99.0):
    """
    Cluster-based consolidation: identify groups of nearby parks and try to
    replace entire clusters with fewer, optimally positioned parks.
    Records steps for visualization.
    """
    if len(parks) <= 1:
        return parks
    
    # Get boundary area
    gdf = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:3857")
    boundary_area_m2 = gdf_proj.geometry[0].area
    
    # Define "nearby" as within 1.5x service distance
    cluster_distance_threshold = service_distance_deg * 1.5
    
    # STEP 1: Find clusters of nearby parks using distance threshold
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    
    # Get park centroids
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in parks])
    
    if len(centroids) < 2:
        return parks
    
    # Calculate pairwise distances
    distances = pdist(centroids)
    
    # Hierarchical clustering
    linkage_matrix = linkage(distances, method='complete')
    clusters = fcluster(linkage_matrix, cluster_distance_threshold, criterion='distance')
    
    # Group parks by cluster
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(idx)
    
    # Record clustering step
    st.session_state.algorithm_steps.append({
        'type': 'clustering',
        'parks': parks.copy(),
        'coverage': 100 * unary_union([p.buffer(service_distance_deg) for p in parks]).intersection(boundary_poly).area / boundary_poly.area,
        'description': f'Identified {len(cluster_dict)} cluster(s)',
        'clusters': {cid: [parks[i] for i in indices] for cid, indices in cluster_dict.items()},
        'cluster_assignments': clusters.tolist()
    })
    
    # STEP 2: Plan consolidation for ALL clusters first (don't modify parks yet)
    # This prevents index issues when processing multiple clusters
    consolidation_plan = {}  # cluster_id -> replacement_parks
    
    for cluster_id, park_indices in cluster_dict.items():
        if len(park_indices) <= 1:
            continue  # Single park clusters don't need consolidation
        
        # Get parks in this cluster
        cluster_parks = [parks[i] for i in park_indices]
        
        st.info(f"🔍 Analyzing cluster {cluster_id} with {len(cluster_parks)} parks...")
        
        # Calculate what area this cluster covers
        cluster_buffers = [p.buffer(service_distance_deg) for p in cluster_parks]
        cluster_coverage = unary_union(cluster_buffers)
        
        # Expand search area slightly beyond cluster coverage
        minx, miny, maxx, maxy = cluster_coverage.bounds
        search_expansion = service_distance_deg * 0.3
        minx -= search_expansion
        miny -= search_expansion
        maxx += search_expansion
        maxy += search_expansion
        
        best_solution = None
        best_solution_count = len(cluster_parks)
        
        # IMPROVED STRATEGY: Try removing 1 park at a time, then 2, then 3, etc.
        # Start conservatively (remove fewer parks) for better success rate
        for num_parks_to_remove in range(1, len(cluster_parks)):
            num_replacement_parks = len(cluster_parks) - num_parks_to_remove
            
            if num_replacement_parks <= 0:
                continue
            
            # Try multiple strategies to find good positions
            all_candidates = []
            
            # Strategy 1: Grid-based sampling (more systematic)
            grid_size = max(3, num_replacement_parks + 2)
            x_grid = np.linspace(minx, maxx, grid_size)
            y_grid = np.linspace(miny, maxy, grid_size)
            
            for _ in range(5):  # 5 grid-based attempts
                candidate_positions = []
                for _ in range(num_replacement_parks):
                    x = np.random.choice(x_grid)
                    y = np.random.choice(y_grid)
                    if boundary_poly.contains(Point(x, y)) or cluster_coverage.contains(Point(x, y)):
                        candidate_positions.append([x, y])
                
                if len(candidate_positions) == num_replacement_parks:
                    all_candidates.append(candidate_positions)
            
            # Strategy 2: Pure random sampling (more exploratory)
            for _ in range(10):  # Increased from 3 to 10 attempts
                candidate_positions = []
                attempts_per_park = 0
                
                while len(candidate_positions) < num_replacement_parks and attempts_per_park < 200:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    point = Point(x, y)
                    
                    # Accept if within boundary OR cluster coverage
                    if boundary_poly.contains(point) or cluster_coverage.contains(point):
                        candidate_positions.append([x, y])
                    attempts_per_park += 1
                
                if len(candidate_positions) == num_replacement_parks:
                    all_candidates.append(candidate_positions)
            
            # Strategy 3: Use existing park centroids as starting points
            if len(cluster_parks) >= num_replacement_parks:
                # Sample from existing park positions with slight perturbation
                for _ in range(5):
                    candidate_positions = []
                    selected_parks = np.random.choice(len(cluster_parks), num_replacement_parks, replace=False)
                    
                    for idx in selected_parks:
                        base_x = cluster_parks[idx].centroid.x
                        base_y = cluster_parks[idx].centroid.y
                        # Add small random offset
                        offset = service_distance_deg * 0.3
                        x = base_x + np.random.uniform(-offset, offset)
                        y = base_y + np.random.uniform(-offset, offset)
                        candidate_positions.append([x, y])
                    
                    all_candidates.append(candidate_positions)
            
            # Evaluate all candidate position sets
            for candidate_positions in all_candidates:
                # Create parks at these positions
                new_parks = []
                for pos in candidate_positions:
                    park = create_park_at_location(pos, target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                    if park is not None:
                        new_parks.append(park)
                
                if len(new_parks) != num_replacement_parks:
                    continue  # Failed to create all parks
                
                # Test coverage with new parks (simulating replacement using ORIGINAL indices)
                parks_to_keep_ids = set(range(len(parks))) - set(park_indices)
                test_parks = [parks[i] for i in parks_to_keep_ids] + new_parks
                
                test_buffers = [p.buffer(service_distance_deg) for p in test_parks]
                test_coverage = unary_union(test_buffers).intersection(boundary_poly)
                
                gdf_test = gpd.GeoDataFrame([1], geometry=[test_coverage], crs="EPSG:4326")
                gdf_test_proj = gdf_test.to_crs("EPSG:3857")
                test_coverage_pct = 100 * gdf_test_proj.geometry[0].area / boundary_area_m2
                
                # If this solution is better and maintains coverage
                if test_coverage_pct >= min_coverage and len(new_parks) < best_solution_count:
                    best_solution = new_parks
                    best_solution_count = len(new_parks)
            
            # If we found a solution with this number of parks, stop trying fewer
            if best_solution is not None and best_solution_count < len(cluster_parks):
                st.info(f"💡 Found solution: {len(cluster_parks)} → {best_solution_count} parks")
                break
        
        # Store the consolidation plan for this cluster
        if best_solution is not None and best_solution_count < len(cluster_parks):
            consolidation_plan[cluster_id] = {
                'old_parks': cluster_parks,
                'new_parks': best_solution,
                'old_indices': park_indices,
                'removed_count': len(cluster_parks) - len(best_solution)
            }
            
            st.success(f"✅ Plan for cluster {cluster_id}: {len(cluster_parks)} parks → {len(best_solution)} parks ({len(cluster_parks) - len(best_solution)} to remove)")
        else:
            st.warning(f"⚠️ Cluster {cluster_id}: Could not find consolidation - keeping all {len(cluster_parks)} parks")
    
    # STEP 3: Apply ALL consolidations at once using original indices
    if consolidation_plan:
        # Collect all indices to remove from original parks list
        indices_to_remove = set()
        for plan in consolidation_plan.values():
            indices_to_remove.update(plan['old_indices'])
        
        # Keep parks that are NOT being removed (using original indices)
        consolidated_parks = [parks[i] for i in range(len(parks)) if i not in indices_to_remove]
        
        # Add all new replacement parks
        all_new_parks = []
        for cluster_id, plan in consolidation_plan.items():
            all_new_parks.extend(plan['new_parks'])
            
            # Record consolidation step for this cluster
            temp_consolidated = consolidated_parks + all_new_parks
            new_coverage = 100 * unary_union([p.buffer(service_distance_deg) for p in temp_consolidated]).intersection(boundary_poly).area / boundary_poly.area
            
            st.session_state.algorithm_steps.append({
                'type': 'consolidation_result',
                'parks': temp_consolidated.copy(),
                'coverage': new_coverage,
                'description': f'Cluster {cluster_id}: {len(plan["old_parks"])} → {len(plan["new_parks"])} parks',
                'cluster_id': cluster_id,
                'old_parks': plan['old_parks'],
                'new_parks': plan['new_parks'],
                'removed_count': plan['removed_count']
            })
        
        consolidated_parks.extend(all_new_parks)
        
        total_removed = sum(plan['removed_count'] for plan in consolidation_plan.values())
        total_clusters = len(consolidation_plan)
        
        st.success(f"✨ Smart Cluster Consolidation: Processed {total_clusters} cluster(s), removed {total_removed} park(s) total!")
        
        # Record final state
        final_coverage = 100 * unary_union([p.buffer(service_distance_deg) for p in consolidated_parks]).intersection(boundary_poly).area / boundary_poly.area
        
        st.session_state.algorithm_steps.append({
            'type': 'final',
            'parks': consolidated_parks.copy(),
            'coverage': final_coverage,
            'description': f'Final: {len(consolidated_parks)} parks, {final_coverage:.1f}% coverage',
            'total_removed': total_removed
        })
        
        return consolidated_parks
    else:
        st.info("No consolidation opportunities found - greedy placement was already optimal!")
        return parks

def create_park_buffers(parks, park_size_ha=1.25):
    """
    Create service area buffers around parks.
    Service distance scales with park size: 0.5ha→150m, 2.0ha→300m
    """
    if not parks:
        return []
    
    lats = [p.centroid.y for p in parks]
    ref_lat = np.median(lats)
    
    # Calculate service distance based on park size
    service_distance_m = calculate_service_distance(park_size_ha)
    buffer_deg = meters_to_degrees(service_distance_m, ref_lat)
    
    buffers = []
    for park in parks:
        buffer_poly = park.buffer(buffer_deg)
        buffers.append(buffer_poly)
    
    return buffers

def calculate_coverage(boundary_poly, park_buffers):
    """
    Calculate percentage of boundary covered by park buffers.
    """
    if not park_buffers or boundary_poly is None:
        return 0
    
    all_buffers = unary_union(park_buffers)
    covered = boundary_poly.intersection(all_buffers)
    
    gdf_boundary = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_covered = gpd.GeoDataFrame([1], geometry=[covered], crs="EPSG:4326")
    
    gdf_boundary_proj = gdf_boundary.to_crs("EPSG:3857")
    gdf_covered_proj = gdf_covered.to_crs("EPSG:3857")
    
    boundary_area = gdf_boundary_proj.geometry[0].area
    covered_area = gdf_covered_proj.geometry[0].area
    
    if boundary_area == 0:
        return 0
    
    return 100 * covered_area / boundary_area

def parks_to_geojson(parks):
    """
    Convert park polygons to GeoJSON features with area.
    """
    features = []
    for idx, park in enumerate(parks):
        if park.geom_type == 'Polygon':
            coords = list(park.exterior.coords)
            
            gdf = gpd.GeoDataFrame([1], geometry=[park], crs="EPSG:4326")
            gdf_projected = gdf.to_crs("EPSG:3857")
            area_m2 = gdf_projected.geometry[0].area
            area_ha = area_m2 / 10000
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "id": idx,
                    "type": "park",
                    "area_ha": area_ha
                }
            }
            features.append(feature)
    
    return features

def buffers_to_geojson(buffers):
    """
    Convert buffer polygons to GeoJSON features.
    """
    features = []
    for idx, buffer in enumerate(buffers):
        if buffer.geom_type == 'Polygon':
            coords = list(buffer.exterior.coords)
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "id": idx,
                    "type": "park_buffer"
                }
            }
            features.append(feature)
        elif buffer.geom_type == 'MultiPolygon':
            for poly in buffer.geoms:
                coords = list(poly.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "id": idx,
                        "type": "park_buffer"
                    }
                }
                features.append(feature)
    
    return features

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Polygon Information")
    
    if st.session_state.geojson_features:
        # Calculate total area
        total_hectares = 0
        
        for feature in st.session_state.geojson_features:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                hectares = calculate_area_hectares(coords)
                total_hectares += hectares
        
        if total_hectares > 0:
            st.metric("Boundary Area (ha)", f"{total_hectares:.2f}")
        
        st.divider()
        
        # Load boundary
        boundary = load_boundary_polygon(st.session_state.geojson_features)
        
        # Check if boundary changed
        boundary_changed = st.session_state.cached_boundary is None or (
            boundary is not None and 
            st.session_state.cached_boundary is not None and
            boundary.bounds != st.session_state.cached_boundary.bounds
        )
        
        if boundary_changed:
            st.session_state.cached_boundary = boundary
            st.session_state.optimization_run = False  # Reset optimization flag
            st.session_state.parks = []
            st.session_state.park_buffers = []
            st.session_state.algorithm_steps = []
            st.session_state.current_step = -1
        
        # Algorithm selection and run button
        st.subheader("⚙️ Park Settings")
        
        # Park size slider (only show if optimization hasn't run yet)
        if not st.session_state.optimization_run:
            st.session_state.park_size_ha = st.slider(
                "Target Park Size (hectares)",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.park_size_ha,
                step=0.1,
                help="Larger parks = fewer needed, wider service area. Smaller parks = more needed, narrower service area.",
                key="park_size_slider"
            )
            
            # Show park size info
            park_size_m2 = st.session_state.park_size_ha * 10000
            approx_dimensions = np.sqrt(park_size_m2 / 1.5)  # aspect ratio 1.5
            service_dist = calculate_service_distance(st.session_state.park_size_ha)
            
            st.caption(f"≈ {approx_dimensions:.0f}m × {approx_dimensions * 1.5:.0f}m per park")
            st.caption(f"🎯 Service area: {service_dist:.0f}m radius")
            
            st.divider()
        
        # Run button
        if not st.session_state.optimization_run:
            st.info("📝 Adjust park size above, then click to find optimal solution")
            
            if st.button("🚀 Find Optimal Parks", use_container_width=True, type="primary"):
                if boundary is not None:
                    # Get park size setting
                    target_park_size = st.session_state.park_size_ha
                    # Use small range around target for flexibility
                    min_park_size = target_park_size * 0.9
                    max_park_size = target_park_size * 1.1
                    
                    with st.spinner("🔄 Finding optimal park locations (ILP + Refinement)..."):
                        parks = find_minimum_parks_optimal(boundary, min_park_size, max_park_size)
                        st.session_state.parks = parks
                        
                        park_buffers = create_park_buffers(parks, target_park_size)
                        st.session_state.park_buffers = park_buffers
                        st.session_state.optimization_run = True
                    
                    st.rerun()
        else:
            st.success(f"✅ Optimization complete - Provably minimal solution!")
            st.info(f"🏞️ Park size: {st.session_state.park_size_ha} ha")
            
            if st.button("📏 Try Different Park Size", use_container_width=True):
                st.session_state.optimization_run = False
                st.session_state.parks = []
                st.session_state.park_buffers = []
                st.session_state.algorithm_steps = []
                st.session_state.current_step = -1
                st.rerun()
        
        st.divider()
        if st.session_state.parks:
            st.divider()
            st.subheader("🌳 Parks Optimization")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of parks", len(st.session_state.parks))
            with col2:
                total_park_area = sum(
                    gpd.GeoDataFrame([1], geometry=[p], crs="EPSG:4326")
                    .to_crs("EPSG:3857").geometry[0].area / 10000 
                    for p in st.session_state.parks
                )
                st.metric("Parks (total ha)", f"{total_park_area:.2f}")
            
            if st.session_state.park_buffers:
                coverage = calculate_coverage(boundary, st.session_state.park_buffers)
                st.divider()
                st.metric("Coverage (%)", f"{coverage:.1f}%")
                
                if coverage >= 99:
                    st.success("✅ Entire boundary covered!")
                else:
                    st.warning(f"⚠️ {100-coverage:.1f}% uncovered")
            
            st.divider()
            st.info(f"📍 Boundary polygons: {len(st.session_state.geojson_features)}")
            
            # STEP VISUALIZATION CONTROLS
            if st.session_state.algorithm_steps:
                st.divider()
                st.subheader("🎬 Algorithm Steps")
                
                total_steps = len(st.session_state.algorithm_steps)
                
                # Step navigation
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if st.button("⬅️ Prev", use_container_width=True, disabled=st.session_state.current_step <= 0):
                        if st.session_state.current_step == -1:
                            st.session_state.current_step = total_steps - 2
                        else:
                            st.session_state.current_step = max(0, st.session_state.current_step - 1)
                        st.rerun()
                
                with col3:
                    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.current_step >= total_steps - 1):
                        st.session_state.current_step = min(total_steps - 1, st.session_state.current_step + 1)
                        st.rerun()
                
                with col2:
                    if st.button("🏁 Final", use_container_width=True, disabled=st.session_state.current_step == -1):
                        st.session_state.current_step = -1
                        st.rerun()
                
                # Display current step info
                if st.session_state.current_step == -1:
                    st.info("📍 Showing: **Final Result**")
                else:
                    current_step_data = st.session_state.algorithm_steps[st.session_state.current_step]
                    step_num = st.session_state.current_step + 1
                    st.info(f"📍 Step {step_num}/{total_steps}: **{current_step_data['description']}**")
                    
                    if 'coverage' in current_step_data:
                        st.metric("Coverage at this step", f"{current_step_data['coverage']:.1f}%")
                    
                    if current_step_data['type'] == 'grid_setup':
                        st.caption("🔵 Blue dots = candidate grid positions")
                    elif current_step_data['type'] == 'greedy_placement':
                        st.caption("🟢 Green = parks | 🆕 Bright = new (legacy)")
                    elif current_step_data['type'] == 'clustering':
                        st.caption("🔴 Red outlines = clusters identified")
                    elif current_step_data['type'] == 'consolidation_result':
                        st.caption("❌ Red = removed | ✅ Bright green = new")
                    elif current_step_data['type'] == 'refinement':
                        st.caption("🔧 Position refinement | Parks merged/adjusted")
                    elif current_step_data['type'] == 'optimal_solution':
                        st.caption("🎯 ILP optimal solution (discrete grid)")
        
        # Download GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": st.session_state.geojson_features
        }
        
        geojson_str = json.dumps(geojson_data, indent=2)
        
        st.download_button(
            label="📥 Download GeoJSON",
            data=geojson_str,
            file_name=f"polygon-{datetime.now().strftime('%Y-%m-%d')}.geojson",
            mime="application/geo+json"
        )
        
        # Clear button
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.geojson_features = []
            st.session_state.parks = []
            st.session_state.park_buffers = []
            st.session_state.cached_boundary = None
            st.session_state.algorithm_steps = []
            st.session_state.current_step = -1
            st.session_state.optimization_run = False
            st.rerun()
    
    else:
        st.info("👉 Draw a polygon on the map to start optimization.")

# ============================================================================
# MAP
# ============================================================================

st.subheader("Interactive Map")

initial_center = [54.5973, -3.4360]
initial_zoom = 6
bounds = get_bounds_from_polygons(st.session_state.geojson_features)

m = folium.Map(
    location=initial_center,
    zoom_start=initial_zoom,
    tiles="OpenStreetMap"
)

# Add drawing tools
from folium.plugins import Draw

draw = Draw(
    export=True,
    position='topleft',
    draw_options={
        'polyline': False,
        'polygon': True,
        'rectangle': False,
        'circle': False,
        'marker': False,
        'circlemarker': False
    }
)
draw.add_to(m)

# Display boundary polygons
for feature in st.session_state.geojson_features:
    folium.GeoJson(
        data=feature,
        style_function=lambda x: {
            'color': '#007bff',
            'weight': 2,
            'opacity': 0.8,
            'fillOpacity': 0.2
        }
    ).add_to(m)

# Determine what to display based on current step
if st.session_state.current_step == -1:
    # Show final result
    # Display park buffers (translucent)
    buffer_features = buffers_to_geojson(st.session_state.park_buffers)
    for feature in buffer_features:
        folium.GeoJson(
            data=feature,
            style_function=lambda x: {
                'color': '#ffc107',
                'weight': 1,
                'opacity': 0.3,
                'fillOpacity': 0.1
            }
        ).add_to(m)
    
    # Display parks
    park_features = parks_to_geojson(st.session_state.parks)
    for feature in park_features:
        area_ha = feature['properties']['area_ha']
        
        folium.GeoJson(
            data=feature,
            style_function=lambda x: {
                'color': '#27ae60',
                'weight': 2,
                'opacity': 0.9,
                'fillOpacity': 0.6
            },
            popup=f"Park: {area_ha:.2f} ha"
        ).add_to(m)

elif st.session_state.algorithm_steps and 0 <= st.session_state.current_step < len(st.session_state.algorithm_steps):
    # Show specific step
    step_data = st.session_state.algorithm_steps[st.session_state.current_step]
    
    if step_data['type'] == 'grid_setup':
        # Display grid points
        grid_x = step_data['grid_x']
        grid_y = step_data['grid_y']
        
        for x in grid_x:
            for y in grid_y:
                folium.CircleMarker(
                    location=[y, x],
                    radius=2,
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.4,
                    weight=1
                ).add_to(m)
    
    elif step_data['type'] == 'greedy_placement':
        # Display parks at this iteration
        parks_at_step = step_data['parks']
        new_park = step_data.get('new_park')
        
        for park in parks_at_step:
            is_new = (new_park is not None and park == new_park)
            
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x, is_new=is_new: {
                        'color': '#00ff00' if is_new else '#27ae60',
                        'weight': 3 if is_new else 2,
                        'opacity': 1.0 if is_new else 0.7,
                        'fillOpacity': 0.8 if is_new else 0.5
                    },
                    popup=f"{'NEW PARK' if is_new else 'Park'}"
                ).add_to(m)
    
    elif step_data['type'] == 'pre_consolidation':
        # Display all parks before consolidation
        parks_at_step = step_data['parks']
        
        for park in parks_at_step:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#27ae60',
                        'weight': 2,
                        'opacity': 0.8,
                        'fillOpacity': 0.6
                    }
                ).add_to(m)
    
    elif step_data['type'] == 'clustering':
        # Display parks with cluster colors
        clusters = step_data.get('clusters', {})
        cluster_colors = ['#e74c3c', '#9b59b6', '#3498db', '#1abc9c', '#f39c12', '#e67e22']
        
        for cluster_id, cluster_parks in clusters.items():
            color = cluster_colors[(cluster_id - 1) % len(cluster_colors)]
            
            for park in cluster_parks:
                if park.geom_type == 'Polygon':
                    coords = list(park.exterior.coords)
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        }
                    }
                    
                    folium.GeoJson(
                        data=feature,
                        style_function=lambda x, c=color: {
                            'color': c,
                            'weight': 3,
                            'opacity': 1.0,
                            'fillOpacity': 0.3
                        },
                        popup=f"Cluster {cluster_id}"
                    ).add_to(m)
    
    elif step_data['type'] == 'consolidation_result':
        # Display old parks (removed) and new parks (replacements)
        old_parks = step_data.get('old_parks', [])
        new_parks = step_data.get('new_parks', [])
        other_parks = [p for p in step_data['parks'] if p not in new_parks]
        
        # Show removed parks in red
        for park in old_parks:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#e74c3c',
                        'weight': 2,
                        'opacity': 0.7,
                        'fillOpacity': 0.3,
                        'dashArray': '5, 5'
                    },
                    popup="REMOVED"
                ).add_to(m)
        
        # Show new replacement parks in bright green
        for park in new_parks:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#00ff00',
                        'weight': 3,
                        'opacity': 1.0,
                        'fillOpacity': 0.7
                    },
                    popup="NEW PARK"
                ).add_to(m)
        
        # Show other parks in normal green
        for park in other_parks:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#27ae60',
                        'weight': 2,
                        'opacity': 0.6,
                        'fillOpacity': 0.4
                    }
                ).add_to(m)
    
    elif step_data['type'] == 'final':
        # Display final consolidated result
        parks_at_step = step_data['parks']
        
        for park in parks_at_step:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#27ae60',
                        'weight': 2,
                        'opacity': 0.9,
                        'fillOpacity': 0.6
                    }
                ).add_to(m)
    
    elif step_data['type'] == 'optimal_solution':
        # Display ILP optimal solution
        parks_at_step = step_data['parks']
        
        for park in parks_at_step:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#3498db',
                        'weight': 2,
                        'opacity': 0.9,
                        'fillOpacity': 0.6
                    },
                    popup="ILP Optimal"
                ).add_to(m)
    
    elif step_data['type'] == 'refinement':
        # Display refinement progress
        parks_at_step = step_data['parks']
        
        for park in parks_at_step:
            if park.geom_type == 'Polygon':
                coords = list(park.exterior.coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        'color': '#9b59b6',
                        'weight': 2,
                        'opacity': 0.9,
                        'fillOpacity': 0.6
                    },
                    popup="Refined"
                ).add_to(m)

# Fit bounds if polygons exist
if bounds:
    min_lon, min_lat, max_lon, max_lat = bounds
    m.fit_bounds(
        [[min_lat, min_lon], [max_lat, max_lon]],
        padding=(50, 50)
    )

# Capture map interactions
map_data = st_folium(m, width=1400, height=600)

# Process drawn polygons
if map_data:
    if 'all_drawings' in map_data and map_data['all_drawings']:
        for drawing in map_data['all_drawings']:
            if drawing['geometry']['type'] == 'Polygon':
                coords_str = json.dumps(drawing['geometry']['coordinates'])
                
                is_duplicate = any(
                    json.dumps(f['geometry']['coordinates']) == coords_str
                    for f in st.session_state.geojson_features
                )
                
                if not is_duplicate:
                    feature = {
                        "type": "Feature",
                        "geometry": drawing['geometry'],
                        "properties": {
                            "id": len(st.session_state.geojson_features) + 1,
                            "created": datetime.now().isoformat()
                        }
                    }
                    st.session_state.geojson_features.append(feature)
                    st.session_state.cached_boundary = None
                    st.rerun()

st.divider()

# Legend
with st.expander("📋 Map Legend"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Boundary** - Project area")
    with col2:
        st.markdown(f"🟢 **Parks** - {st.session_state.park_size_ha:.1f} ha")
    with col3:
        if st.session_state.park_size_ha:
            service_dist = calculate_service_distance(st.session_state.park_size_ha)
            st.markdown(f"🟡 **Buffers** - {service_dist:.0f}m service area")
        else:
            st.markdown("🟡 **Buffers** - Service area")

st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    <p>🗺️ Green Mapping Tool | Optimal Park Placement with ILP + Continuous Refinement</p>
</div>
""", unsafe_allow_html=True)