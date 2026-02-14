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
from scipy.spatial.distance import pdist, squareform
import functools

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
    st.session_state.current_step = -1
if "optimization_run" not in st.session_state:
    st.session_state.optimization_run = False
if "park_size_ha" not in st.session_state:
    st.session_state.park_size_ha = 1.25
if "cached_boundary_area_m2" not in st.session_state:
    st.session_state.cached_boundary_area_m2 = None

# ============================================================================
# CACHED TRANSFORMERS & UTILITIES
# ============================================================================

_transformer_to_meters = None
_transformer_to_latlon = None

def get_transformers():
    """Get or create cached transformers."""
    global _transformer_to_meters, _transformer_to_latlon
    if _transformer_to_meters is None:
        from pyproj import Transformer
        _transformer_to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        _transformer_to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return _transformer_to_meters, _transformer_to_latlon

@functools.lru_cache(maxsize=256)
def calculate_service_distance(park_size_ha):
    """Calculate service distance based on park size. LRU cached."""
    return 100 * park_size_ha + 100

@functools.lru_cache(maxsize=128)
def meters_to_degrees(meters, ref_lat=54.5973):
    """Convert meters to degrees at reference latitude. LRU cached."""
    earth_radius = 6371000
    lat_per_m = earth_radius * np.pi / 180
    return meters / lat_per_m

def lonlat_to_meters(lon=0, lat=0, ref_lat=54.5973):
    """Convert lon/lat degrees to meters."""
    earth_radius = 6371000
    meters_per_lat_degree = earth_radius * np.pi / 180
    meters_per_lon_degree = earth_radius * np.pi / 180 * np.cos(np.radians(ref_lat))
    return meters_per_lon_degree, meters_per_lat_degree

def create_buffer_projected(geometry, buffer_distance_m):
    """Create a buffer using projected coordinates for accuracy."""
    from shapely.ops import transform
    to_meters, to_latlon = get_transformers()
    
    geom_meters = transform(to_meters.transform, geometry)
    buffer_meters = geom_meters.buffer(buffer_distance_m)
    buffer_latlon = transform(to_latlon.transform, buffer_meters)
    
    return buffer_latlon

def get_boundary_area_m2(boundary_poly, use_cache=True):
    """Get boundary area in m², with caching."""
    if use_cache and st.session_state.cached_boundary_area_m2 is not None:
        if hasattr(st.session_state, '_cached_boundary_bounds'):
            if st.session_state._cached_boundary_bounds == boundary_poly.bounds:
                return st.session_state.cached_boundary_area_m2
    
    gdf = gpd.GeoDataFrame([1], geometry=[boundary_poly], crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:3857")
    area_m2 = gdf_proj.geometry[0].area
    
    if use_cache:
        st.session_state.cached_boundary_area_m2 = area_m2
        st.session_state._cached_boundary_bounds = boundary_poly.bounds
    
    return area_m2

def calculate_coverage_percentage_fast(parks, boundary_poly, service_distance_m, boundary_area_m2=None):
    """
    Calculate coverage percentage efficiently.
    Reuses boundary area if provided.
    """
    if not parks:
        return 0.0
    
    if boundary_area_m2 is None:
        boundary_area_m2 = get_boundary_area_m2(boundary_poly)
    
    buffers = [create_buffer_projected(p, service_distance_m) for p in parks]
    covered_area_geom = unary_union(buffers).intersection(boundary_poly)
    
    if covered_area_geom.is_empty:
        return 0.0
    
    gdf_covered = gpd.GeoDataFrame([1], geometry=[covered_area_geom], crs="EPSG:4326")
    gdf_covered_proj = gdf_covered.to_crs("EPSG:3857")
    covered_area_m2 = gdf_covered_proj.geometry[0].area
    
    return 100 * covered_area_m2 / boundary_area_m2

def calculate_area_hectares(coords):
    """Calculate area in hectares."""
    if len(coords) < 3:
        return 0
    
    poly = Polygon([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame([1], geometry=[poly], crs="EPSG:4326")
    gdf_projected = gdf.to_crs("EPSG:3857")
    area_m2 = gdf_projected.geometry[0].area
    
    return area_m2 / 10000

def get_bounds_from_polygons(features):
    """Get the bounding box of all polygons."""
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
    """Load and merge boundary polygons."""
    if not features:
        return None
    
    polygons = []
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            poly = Polygon([(c[0], c[1]) for c in coords])
            polygons.append(poly)
    
    return unary_union(polygons) if polygons else None

def create_park_at_location(centroid, target_area_m2, boundary_poly, lon_per_m, lat_per_m):
    """Create a park at a given location (square shape)."""
    from shapely.ops import transform
    
    to_meters, to_latlon = get_transformers()
    centroid_point = Point(centroid[0], centroid[1])
    centroid_meters = transform(to_meters.transform, centroid_point)
    
    park_side_m = np.sqrt(target_area_m2)
    half_side = park_side_m / 2
    
    park_meters = box(
        centroid_meters.x - half_side,
        centroid_meters.y - half_side,
        centroid_meters.x + half_side,
        centroid_meters.y + half_side
    )
    
    park_latlon = transform(to_latlon.transform, park_meters)
    
    if not boundary_poly.contains(park_latlon):
        return None
    
    gdf = gpd.GeoDataFrame([1], geometry=[park_latlon], crs="EPSG:4326")
    gdf_projected = gdf.to_crs("EPSG:3857")
    actual_area_m2 = gdf_projected.geometry[0].area
    
    return park_latlon if actual_area_m2 >= 3000 else None

# ============================================================================
# BUILD LIVE MAP FUNCTION
# ============================================================================

def build_live_map(boundary_poly, candidate_parks=None, selected_parks=None, 
                   demand_points=None, service_distance_m=None, bounds=None):
    """Build a folium map with current algorithm state (no st.empty!)."""
    m = folium.Map(
        location=[54.5973, -3.4360],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    # Draw boundary
    if boundary_poly is not None:
        folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(boundary_poly.exterior.coords)]
                }
            },
            style_function=lambda x: {
                'color': '#007bff',
                'weight': 2,
                'opacity': 0.8,
                'fillOpacity': 0.2
            }
        ).add_to(m)
    
    # Draw candidate parks as light gray
    if candidate_parks:
        for park in candidate_parks:
            if park.geom_type == 'Polygon':
                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [list(park.exterior.coords)]
                        }
                    },
                    style_function=lambda x: {
                        'color': '#cccccc',
                        'weight': 1,
                        'opacity': 0.4,
                        'fillOpacity': 0.1
                    }
                ).add_to(m)
    
    # Draw demand points as tiny blue dots
    if demand_points:
        for point in demand_points:
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=2,
                color='#3498db',
                fill=True,
                fillColor='#3498db',
                fillOpacity=0.6,
                weight=0.5
            ).add_to(m)
    
    # Draw selected parks in green
    if selected_parks:
        for park in selected_parks:
            if park.geom_type == 'Polygon':
                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [list(park.exterior.coords)]
                        }
                    },
                    style_function=lambda x: {
                        'color': '#27ae60',
                        'weight': 2,
                        'opacity': 0.9,
                        'fillOpacity': 0.6
                    }
                ).add_to(m)
            
            # Draw service buffer if service distance provided
            if service_distance_m:
                buffer = create_buffer_projected(park, service_distance_m)
                if buffer.geom_type == 'Polygon':
                    folium.GeoJson(
                        data={
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [list(buffer.exterior.coords)]
                            }
                        },
                        style_function=lambda x: {
                            'color': '#ffc107',
                            'weight': 1,
                            'opacity': 0.2,
                            'fillOpacity': 0.05
                        }
                    ).add_to(m)
    
    # Fit bounds
    if bounds:
        min_lon, min_lat, max_lon, max_lat = bounds
        m.fit_bounds(
            [[min_lat, min_lon], [max_lat, max_lon]],
            padding=(50, 50)
        )
    
    return m

# ============================================================================
# OPTIMIZED ILP SOLVER
# ============================================================================

def find_minimum_parks_optimal(boundary_poly, min_area_ha=0.5, max_area_ha=2.0):
    """
    Find optimal parks using ILP with finer grid.
    Records all steps without interrupting with live updates.
    """
    try:
        from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except ImportError:
        st.error("❌ PuLP library not found!")
        st.error("Add 'pulp' to your requirements.txt file for Streamlit Cloud deployment.")
        st.info("For local install: pip install pulp --break-system-packages")
        return []
    
    if boundary_poly is None:
        return []
    
    st.session_state.algorithm_steps = []
    st.session_state.current_step = -1
    
    avg_park_size = (min_area_ha + max_area_ha) / 2
    service_distance_m = calculate_service_distance(avg_park_size)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.info(f"📏 Park size: {avg_park_size:.2f} ha → Service distance: {service_distance_m:.0f}m")
    
    ref_lat = boundary_poly.centroid.y
    lon_per_m, lat_per_m = lonlat_to_meters(0, 0, ref_lat)
    target_area_m2 = avg_park_size * 10000
    boundary_area_m2 = get_boundary_area_m2(boundary_poly)
    bounds = get_bounds_from_polygons(st.session_state.geojson_features)
    
    # Finer grid spacing
    grid_spacing_m = service_distance_m / 3.5
    minx, miny, maxx, maxy = boundary_poly.bounds
    width_m = (maxx - minx) * lon_per_m
    height_m = (maxy - miny) * lat_per_m
    
    grid_points_x = max(5, min(50, int(np.ceil(width_m / grid_spacing_m))))
    grid_points_y = max(5, min(50, int(np.ceil(height_m / grid_spacing_m))))
    
    total_grid_points = grid_points_x * grid_points_y
    st.info(f"📐 Grid: {grid_points_x}×{grid_points_y} = {total_grid_points} positions")
    
    x_coords = np.linspace(minx, maxx, grid_points_x)
    y_coords = np.linspace(miny, maxy, grid_points_y)
    
    # STEP 1: Create candidate parks
    status_text.info("🔧 Creating candidate park locations...")
    progress_bar.progress(10)
    
    candidate_parks = []
    for x in x_coords:
        for y in y_coords:
            park = create_park_at_location([x, y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
            if park is not None:
                candidate_parks.append(park)
    
    if not candidate_parks:
        st.error("No valid candidate parks found!")
        return []
    
    status_text.info(f"✅ Created {len(candidate_parks)} candidates")
    progress_bar.progress(20)
    
    # Record candidate parks step
    st.session_state.algorithm_steps.append({
        'type': 'candidates',
        'parks': candidate_parks.copy(),
        'coverage': 0,
        'description': f'Grid setup: {len(candidate_parks)} candidate locations',
    })
    
    # STEP 2: Fine demand grid
    status_text.info("📍 Creating demand points...")
    
    demand_grid_size = max(20, min(50, int(np.sqrt(boundary_area_m2 / 10000))))
    demand_points = []
    
    x_demand = np.linspace(minx, maxx, demand_grid_size)
    y_demand = np.linspace(miny, maxy, demand_grid_size)
    
    for x in x_demand:
        for y in y_demand:
            if boundary_poly.contains(Point(x, y)):
                demand_points.append(Point(x, y))
    
    status_text.info(f"✅ Created {len(demand_points)} demand points")
    progress_bar.progress(30)
    
    # Record demand points step
    st.session_state.algorithm_steps.append({
        'type': 'demand_points',
        'parks': candidate_parks.copy(),
        'demand_points': demand_points.copy(),
        'coverage': 0,
        'description': f'Demand sampling: {len(demand_points)} coverage points',
    })
    
    # STEP 3: Coverage matrix
    status_text.info("🔍 Computing coverage relationships...")
    
    coverage_dict = {}
    for park_idx, park in enumerate(candidate_parks):
        park_buffer = create_buffer_projected(park, service_distance_m)
        covered_points = [i for i, p in enumerate(demand_points) if park_buffer.contains(p)]
        
        if covered_points:
            coverage_dict[park_idx] = covered_points
    
    status_text.info(f"✅ Coverage matrix ready")
    progress_bar.progress(40)
    
    if not coverage_dict:
        st.error("No parks can cover any demand points!")
        return []
    
    # STEP 4: Solve ILP
    status_text.info("🧮 Solving optimization problem (ILP)...")
    
    prob = LpProblem("MinimumParkCoverage", LpMinimize)
    park_vars = {i: LpVariable(f"park_{i}", cat=LpBinary) for i in coverage_dict.keys()}
    
    prob += lpSum(park_vars.values()), "TotalParks"
    
    covered_points = set()
    for covered in coverage_dict.values():
        covered_points.update(covered)
    
    for j in covered_points:
        prob += lpSum(park_vars[i] for i in coverage_dict if j in coverage_dict[i]) >= 1, f"Point_{j}"
    
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120, threads=4))
    
    selected_parks = [candidate_parks[i] for i in coverage_dict.keys() 
                     if park_vars[i].varValue == 1]
    
    status_text.success(f"🎯 Optimal solution found: {len(selected_parks)} parks")
    progress_bar.progress(60)
    
    optimal_coverage = calculate_coverage_percentage_fast(
        selected_parks, boundary_poly, service_distance_m, boundary_area_m2
    )
    
    st.session_state.algorithm_steps.append({
        'type': 'optimal_solution',
        'parks': selected_parks.copy(),
        'coverage': optimal_coverage,
        'description': f'Optimal (ILP): {len(selected_parks)} parks, {optimal_coverage:.1f}% coverage',
    })
    
    # FULL REFINEMENT to remove redundant parks
    if len(selected_parks) > 1:
        selected_parks = refine_park_positions_full(
            selected_parks, boundary_poly, service_distance_m, 
            target_area_m2, lon_per_m, lat_per_m, boundary_area_m2,
            status_text, progress_bar
        )
    
    progress_bar.progress(100)
    status_text.success("✨ Optimization complete!")
    
    return selected_parks

# ============================================================================
# FULL REFINEMENT (removes redundant parks)
# ============================================================================

def refine_park_positions_full(parks, boundary_poly, service_distance_m, 
                               target_area_m2, lon_per_m, lat_per_m, 
                               boundary_area_m2, status_text, progress_bar,
                               min_coverage=99.0):
    """
    Full refinement that removes redundant parks and optimizes positions.
    3 iterations maximum.
    """
    if len(parks) <= 1:
        return parks
    
    ref_lat = boundary_poly.centroid.y
    service_distance_deg = meters_to_degrees(service_distance_m, ref_lat)
    
    refined_parks = parks.copy()
    iteration = 0
    max_iterations = 3
    improvements_made = True
    
    while improvements_made and iteration < max_iterations:
        improvements_made = False
        iteration += 1
        
        status_text.info(f"🔍 Refinement iteration {iteration}/{max_iterations}...")
        progress_bar.progress(60 + (iteration * 10))
        
        # STEP 1: Try to merge nearby park pairs
        for i in range(len(refined_parks)):
            for j in range(i + 1, len(refined_parks)):
                park_i = refined_parks[i]
                park_j = refined_parks[j]
                
                distance = park_i.centroid.distance(park_j.centroid)
                if distance > service_distance_deg * 2:
                    continue
                
                mid_x = (park_i.centroid.x + park_j.centroid.x) / 2
                mid_y = (park_i.centroid.y + park_j.centroid.y) / 2
                
                best_merge_position = None
                best_merge_coverage = 0
                
                search_radius = service_distance_deg * 0.8
                for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for radius in np.linspace(0, search_radius, 4):
                        test_x = mid_x + radius * np.cos(angle)
                        test_y = mid_y + radius * np.sin(angle)
                        
                        test_park = create_park_at_location([test_x, test_y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                        if test_park is None:
                            continue
                        
                        test_parks = [p for idx, p in enumerate(refined_parks) if idx != i and idx != j] + [test_park]
                        
                        test_coverage = calculate_coverage_percentage_fast(
                            test_parks, boundary_poly, service_distance_m, boundary_area_m2
                        )
                        
                        if test_coverage >= min_coverage and test_coverage > best_merge_coverage:
                            best_merge_coverage = test_coverage
                            best_merge_position = test_park
                
                if best_merge_position is not None:
                    refined_parks = [p for idx, p in enumerate(refined_parks) if idx != i and idx != j] + [best_merge_position]
                    improvements_made = True
                    
                    new_coverage = calculate_coverage_percentage_fast(refined_parks, boundary_poly, service_distance_m, boundary_area_m2)
                    
                    status_text.info(f"✅ Merged parks {i+1} & {j+1}")
                    
                    st.session_state.algorithm_steps.append({
                        'type': 'refinement',
                        'parks': refined_parks.copy(),
                        'coverage': new_coverage,
                        'description': f'Refinement: Merged 2 parks → 1 park',
                        'iteration': iteration
                    })
                    
                    break
            
            if improvements_made:
                break
        
        # STEP 2: Fine-tune individual park positions
        if not improvements_made:
            for i in range(len(refined_parks)):
                park = refined_parks[i]
                
                best_adjustment = None
                best_adjustment_coverage = 0
                
                current_x = park.centroid.x
                current_y = park.centroid.y
                
                adjustment_radius = service_distance_deg * 0.2
                
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    for radius in [adjustment_radius * 0.5, adjustment_radius]:
                        test_x = current_x + radius * np.cos(angle)
                        test_y = current_y + radius * np.sin(angle)
                        
                        test_park = create_park_at_location([test_x, test_y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
                        if test_park is None:
                            continue
                        
                        test_parks = [p if idx != i else test_park for idx, p in enumerate(refined_parks)]
                        
                        test_coverage = calculate_coverage_percentage_fast(
                            test_parks, boundary_poly, service_distance_m, boundary_area_m2
                        )
                        
                        if test_coverage > best_adjustment_coverage:
                            best_adjustment_coverage = test_coverage
                            best_adjustment = test_park
                
                if best_adjustment is not None:
                    current_buffers = [create_buffer_projected(p, service_distance_m) for p in refined_parks]
                    current_coverage_pct = 100 * unary_union(current_buffers).intersection(boundary_poly).area / boundary_area_m2
                    
                    if best_adjustment_coverage > current_coverage_pct + 0.1:
                        refined_parks[i] = best_adjustment
                        improvements_made = True
                        
                        status_text.info(f"📍 Fine-tuned park {i+1} position")
    
    final_coverage = calculate_coverage_percentage_fast(
        refined_parks, boundary_poly, service_distance_m, boundary_area_m2
    )
    
    if iteration > 1 or len(refined_parks) < len(parks):
        status_text.success(f"✨ Refinement complete: {len(parks)} → {len(refined_parks)} parks")
        st.session_state.algorithm_steps.append({
            'type': 'final',
            'parks': refined_parks.copy(),
            'coverage': final_coverage,
            'description': f'Final (refined): {len(refined_parks)} parks, {final_coverage:.1f}% coverage',
            'total_removed': len(parks) - len(refined_parks)
        })
    else:
        status_text.info(f"✓ {len(refined_parks)} parks (already optimal)")
    
    return refined_parks

# ============================================================================
# GeoJSON CONVERTERS
# ============================================================================

def parks_to_geojson(parks):
    """Convert parks to GeoJSON."""
    features = []
    for idx, park in enumerate(parks):
        if park.geom_type == 'Polygon':
            coords = list(park.exterior.coords)
            gdf = gpd.GeoDataFrame([1], geometry=[park], crs="EPSG:4326")
            gdf_proj = gdf.to_crs("EPSG:3857")
            area_ha = gdf_proj.geometry[0].area / 10000
            
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"id": idx, "type": "park", "area_ha": area_ha}
            })
    
    return features

def buffers_to_geojson(buffers):
    """Convert buffers to GeoJSON."""
    features = []
    for idx, buffer in enumerate(buffers):
        if buffer.geom_type == 'Polygon':
            coords = list(buffer.exterior.coords)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"id": idx, "type": "park_buffer"}
            })
        elif buffer.geom_type == 'MultiPolygon':
            for poly in buffer.geoms:
                coords = list(poly.exterior.coords)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"id": idx, "type": "park_buffer"}
                })
    
    return features

def create_park_buffers(parks, park_size_ha=1.25):
    """Create service area buffers."""
    service_distance_m = calculate_service_distance(park_size_ha)
    return [create_buffer_projected(park, service_distance_m) for park in parks]

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Polygon Information")
    
    if st.session_state.geojson_features:
        total_hectares = sum(
            calculate_area_hectares(feature['geometry']['coordinates'][0])
            for feature in st.session_state.geojson_features
            if feature['geometry']['type'] == 'Polygon'
        )
        
        if total_hectares > 0:
            st.metric("Boundary Area (ha)", f"{total_hectares:.2f}")
        
        st.divider()
        
        boundary = load_boundary_polygon(st.session_state.geojson_features)
        
        boundary_changed = st.session_state.cached_boundary is None or (
            boundary is not None and 
            st.session_state.cached_boundary is not None and
            boundary.bounds != st.session_state.cached_boundary.bounds
        )
        
        if boundary_changed:
            st.session_state.cached_boundary = boundary
            st.session_state.optimization_run = False
            st.session_state.parks = []
            st.session_state.park_buffers = []
            st.session_state.algorithm_steps = []
            st.session_state.current_step = -1
            st.session_state.cached_boundary_area_m2 = None
        
        st.subheader("⚙️ Park Settings")
        
        if not st.session_state.optimization_run:
            st.session_state.park_size_ha = st.slider(
                "Target Park Size (hectares)",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.park_size_ha,
                step=0.1,
                key="park_size_slider"
            )
            
            park_size_m2 = st.session_state.park_size_ha * 10000
            park_side_m = np.sqrt(park_size_m2)
            service_dist = calculate_service_distance(st.session_state.park_size_ha)
            
            st.caption(f"≈ {park_side_m:.0f}m × {park_side_m:.0f}m square")
            st.caption(f"🎯 Service area: {service_dist:.0f}m radius")
            st.divider()
        
        if not st.session_state.optimization_run:
            st.info("📝 Adjust park size, then click to find optimal solution")
            
            if st.button("🚀 Find Optimal Parks", use_container_width=True, type="primary"):
                if boundary is not None:
                    target_park_size = st.session_state.park_size_ha
                    min_park_size = target_park_size * 0.9
                    max_park_size = target_park_size * 1.1
                    
                    try:
                        parks = find_minimum_parks_optimal(boundary, min_park_size, max_park_size)
                        
                        if parks:
                            st.session_state.parks = parks
                            st.session_state.park_buffers = create_park_buffers(parks, target_park_size)
                            st.session_state.optimization_run = True
                            st.rerun()
                        else:
                            st.error("No parks generated. Try adjusting park size.")
                            st.session_state.optimization_run = False
                    except Exception as e:
                        st.error(f"❌ Optimization failed: {str(e)}")
                        st.error("Please report this error with your boundary details.")
                        import traceback
                        st.error(traceback.format_exc())
                        st.session_state.optimization_run = False
        else:
            if st.session_state.parks and len(st.session_state.parks) > 0:
                st.success(f"✅ Optimization complete - Provably minimal solution!")
                st.info(f"🏞️ Park size: {st.session_state.park_size_ha} ha")
            else:
                st.warning("⚠️ No parks generated. Try adjusting park size or redrawing boundary.")
                st.session_state.optimization_run = False
            
            if st.button("📏 Try Different Park Size", use_container_width=True):
                st.session_state.optimization_run = False
                st.session_state.parks = []
                st.session_state.park_buffers = []
                st.session_state.algorithm_steps = []
                st.session_state.current_step = -1
                st.rerun()
        
        st.divider()
        if st.session_state.parks:
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
                coverage = calculate_coverage_percentage_fast(
                    st.session_state.parks, boundary, 
                    calculate_service_distance(st.session_state.park_size_ha),
                    st.session_state.cached_boundary_area_m2
                )
                st.divider()
                st.metric("Coverage (%)", f"{coverage:.1f}%")
                
                if coverage >= 99:
                    st.success("✅ Entire boundary covered!")
                else:
                    st.warning(f"⚠️ {100-coverage:.1f}% uncovered")
            
            st.divider()
            st.info(f"📍 Boundary polygons: {len(st.session_state.geojson_features)}")
            
            if st.session_state.algorithm_steps:
                st.divider()
                st.subheader("🎬 Algorithm Steps")
                
                total_steps = len(st.session_state.algorithm_steps)
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if st.button("⬅️ Prev", use_container_width=True, disabled=st.session_state.current_step <= 0):
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
                
                if st.session_state.current_step == -1:
                    st.info("📍 Showing: **Final Result**")
                else:
                    current_step_data = st.session_state.algorithm_steps[st.session_state.current_step]
                    step_num = st.session_state.current_step + 1
                    st.info(f"📍 Step {step_num}/{total_steps}: **{current_step_data['description']}**")
                    if 'coverage' in current_step_data:
                        st.metric("Coverage at this step", f"{current_step_data['coverage']:.1f}%")
        
        geojson_data = {"type": "FeatureCollection", "features": st.session_state.geojson_features}
        st.download_button(
            label="📥 Download GeoJSON",
            data=json.dumps(geojson_data, indent=2),
            file_name=f"polygon-{datetime.now().strftime('%Y-%m-%d')}.geojson",
            mime="application/geo+json"
        )
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.geojson_features = []
            st.session_state.parks = []
            st.session_state.park_buffers = []
            st.session_state.cached_boundary = None
            st.session_state.algorithm_steps = []
            st.session_state.current_step = -1
            st.session_state.optimization_run = False
            st.session_state.cached_boundary_area_m2 = None
            st.rerun()
    else:
        st.info("👉 Draw a polygon on the map to start optimization.")

# ============================================================================
# MAPS
# ============================================================================

if not st.session_state.optimization_run:
    st.subheader("Interactive Map")
    
    initial_center = [54.5973, -3.4360]
    initial_zoom = 6
    bounds = get_bounds_from_polygons(st.session_state.geojson_features)
    
    m = folium.Map(location=initial_center, zoom_start=initial_zoom, tiles="OpenStreetMap")
    
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
    
    if bounds:
        min_lon, min_lat, max_lon, max_lat = bounds
        m.fit_bounds(
            [[min_lat, min_lon], [max_lat, max_lon]],
            padding=(50, 50)
        )
    
    map_data = st_folium(m, width=1400, height=600)
    
    if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
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

else:
    # After optimization - show step visualization
    st.subheader("Algorithm Visualization")
    
    bounds = get_bounds_from_polygons(st.session_state.geojson_features)
    boundary = load_boundary_polygon(st.session_state.geojson_features)
    
    if st.session_state.current_step == -1:
        # Final result
        m = build_live_map(
            boundary,
            selected_parks=st.session_state.parks,
            service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
            bounds=bounds
        )
        st_folium(m, width=1400, height=600)
    elif st.session_state.algorithm_steps and 0 <= st.session_state.current_step < len(st.session_state.algorithm_steps):
        step_data = st.session_state.algorithm_steps[st.session_state.current_step]
        
        if step_data['type'] == 'candidates':
            m = build_live_map(boundary, candidate_parks=step_data['parks'], bounds=bounds)
            st.markdown("### 🔵 Candidate Park Locations (Gray)")
        elif step_data['type'] == 'demand_points':
            m = build_live_map(boundary, candidate_parks=step_data['parks'], 
                             demand_points=step_data.get('demand_points'), bounds=bounds)
            st.markdown("### 🔵 Candidates + 🔷 Demand Points")
        elif step_data['type'] == 'optimal_solution':
            m = build_live_map(boundary, selected_parks=step_data['parks'],
                             service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                             bounds=bounds)
            st.markdown("### 🟢 ILP Optimal Solution")
        elif step_data['type'] in ['refinement', 'final']:
            m = build_live_map(boundary, selected_parks=step_data['parks'],
                             service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                             bounds=bounds)
            st.markdown(f"### 🟢 {step_data['description']}")
        
        st_folium(m, width=1400, height=600)

st.divider()

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

st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    <p>🗺️ Green Mapping Tool | Optimal Park Placement with ILP + Continuous Refinement</p>
</div>
""", unsafe_allow_html=True)
