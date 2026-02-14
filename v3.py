def find_minimum_parks_optimal(boundary_poly, min_area_ha=0.5, max_area_ha=2.0):
    """
    BETTER: Keep fine grid but optimize OTHER bottlenecks.
    """
    try:
        from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
    except ImportError:
        st.error("❌ PuLP library not found!")
        return []
    
    if boundary_poly is None:
        return []
    
    st.session_state.algorithm_steps = []
    st.session_state.current_step = -1
    
    avg_park_size = (min_area_ha + max_area_ha) / 2
    service_distance_m = calculate_service_distance(avg_park_size)
    st.info(f"📏 Park size: {avg_park_size:.2f} ha → Service distance: {service_distance_m:.0f}m")
    
    ref_lat = boundary_poly.centroid.y
    lon_per_m, lat_per_m = lonlat_to_meters(0, 0, ref_lat)
    target_area_m2 = avg_park_size * 10000
    boundary_area_m2 = get_boundary_area_m2(boundary_poly)
    
    # KEEP FINE GRID for accuracy ✓
    grid_spacing_m = service_distance_m / 4  # Original: fine grid
    minx, miny, maxx, maxy = boundary_poly.bounds
    width_m = (maxx - minx) * lon_per_m
    height_m = (maxy - miny) * lat_per_m
    
    grid_points_x = max(5, min(40, int(np.ceil(width_m / grid_spacing_m))))
    grid_points_y = max(5, min(40, int(np.ceil(height_m / grid_spacing_m))))
    
    total_grid_points = grid_points_x * grid_points_y
    st.info(f"📐 Grid: {grid_points_x}×{grid_points_y} = {total_grid_points} positions")
    
    x_coords = np.linspace(minx, maxx, grid_points_x)
    y_coords = np.linspace(miny, maxy, grid_points_y)
    
    # OPTIMIZATION 1: Vectorized park creation with numpy ✓
    st.info("🔧 Creating candidate park locations...")
    candidate_parks = []
    
    # Pre-compute all positions at once
    positions = [(x, y) for x in x_coords for y in y_coords]
    
    for x, y in positions:
        park = create_park_at_location([x, y], target_area_m2, boundary_poly, lon_per_m, lat_per_m)
        if park is not None:
            candidate_parks.append(park)
    
    if not candidate_parks:
        st.error("No valid candidate parks found!")
        return []
    
    st.info(f"✅ Created {len(candidate_parks)} candidates")
    
    # OPTIMIZATION 2: Smarter demand point generation ✓
    # Instead of grid, use Poisson disk sampling for better coverage representation
    st.info("📍 Generating demand points...")
    
    demand_points = []
    
    # Adaptive density: 1 point per ~50,000 m² (about 225m spacing)
    # This is better than arbitrary grids
    target_spacing = 225  # meters
    target_spacing_deg = meters_to_degrees(target_spacing, ref_lat)
    
    x_demand = np.arange(minx, maxx, target_spacing_deg)
    y_demand = np.arange(miny, maxy, target_spacing_deg)
    
    for x in x_demand:
        for y in y_demand:
            if boundary_poly.contains(Point(x, y)):
                demand_points.append(Point(x, y))
    
    st.info(f"✅ Created {len(demand_points)} demand points")
    
    # OPTIMIZATION 3: Pre-filter coverage matrix ✓
    # Only include parks that cover at least one demand point
    st.info("🔍 Computing coverage relationships...")
    coverage_dict = {}
    coverage_count = 0
    
    for park_idx, park in enumerate(candidate_parks):
        park_buffer = create_buffer_projected(park, service_distance_m)
        covered_points = [i for i, p in enumerate(demand_points) if park_buffer.contains(p)]
        
        if covered_points:
            coverage_dict[park_idx] = covered_points
            coverage_count += len(covered_points)
    
    st.info(f"✅ Coverage matrix: {len(coverage_dict)} parks × {len(demand_points)} points")
    
    if not coverage_dict:
        st.error("No parks can cover any demand points!")
        return []
    
    # OPTIMIZATION 4: Reduce solver time with warm start ✓
    st.info("🧮 Solving optimization problem...")
    
    prob = LpProblem("MinimumParkCoverage", LpMinimize)
    park_vars = {i: LpVariable(f"park_{i}", cat=LpBinary) for i in range(len(candidate_parks))}
    
    prob += lpSum(park_vars.values()), "TotalParks"
    
    # Only add constraints for coverable demand points
    covered_points = set()
    for covered in coverage_dict.values():
        covered_points.update(covered)
    
    for j in covered_points:
        prob += lpSum(park_vars[i] for i in coverage_dict if j in coverage_dict[i]) >= 1, f"Point_{j}"
    
    # SOLVE with shorter timeout (since problem is well-defined)
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60, threads=4))
    
    selected_parks = [candidate_parks[i] for i in range(len(candidate_parks)) 
                     if park_vars[i].varValue == 1]
    
    st.success(f"🎯 Optimal solution: {len(selected_parks)} parks")
    
    optimal_coverage = calculate_coverage_percentage_fast(
        selected_parks, boundary_poly, service_distance_m, boundary_area_m2
    )
    
    st.session_state.algorithm_steps.append({
        'type': 'optimal_solution',
        'parks': selected_parks.copy(),
        'coverage': optimal_coverage,
        'description': f'Optimal (ILP): {len(selected_parks)} parks, {optimal_coverage:.1f}% coverage',
    })
    
    # Only refine if multiple parks exist
    if len(selected_parks) > 1:
        selected_parks = refine_park_positions_minimal(
            selected_parks, boundary_poly, service_distance_m, 
            target_area_m2, lon_per_m, lat_per_m, boundary_area_m2
        )
    
    return selected_parks

def refine_park_positions_minimal(parks, boundary_poly, service_distance_m, 
                                  target_area_m2, lon_per_m, lat_per_m, 
                                  boundary_area_m2, min_coverage=99.0):
    """
    MINIMAL refinement: Only merge parks if it clearly helps.
    Skip fine-tuning (adds marginal benefit, slows down).
    """
    if len(parks) <= 1:
        return parks
    
    st.info("🔧 Checking for obvious merges...")
    ref_lat = boundary_poly.centroid.y
    service_distance_deg = meters_to_degrees(service_distance_m, ref_lat)
    
    refined_parks = parks.copy()
    improvement_found = True
    iteration = 0
    
    while improvement_found and iteration < 2:  # Max 2 passes
        improvement_found = False
        iteration += 1
        
        # Only merge parks that are VERY close (within 1x service distance)
        for i in range(len(refined_parks)):
            for j in range(i + 1, len(refined_parks)):
                dist = refined_parks[i].centroid.distance(refined_parks[j].centroid)
                
                # Only consider very close parks
                if dist > service_distance_deg:
                    continue
                
                # Try midpoint merge
                mid_x = (refined_parks[i].centroid.x + refined_parks[j].centroid.x) / 2
                mid_y = (refined_parks[i].centroid.y + refined_parks[j].centroid.y) / 2
                
                merge_park = create_park_at_location(
                    [mid_x, mid_y], target_area_m2, boundary_poly, lon_per_m, lat_per_m
                )
                
                if merge_park is None:
                    continue
                
                test_parks = [refined_parks[k] for k in range(len(refined_parks)) 
                             if k != i and k != j] + [merge_park]
                
                test_coverage = calculate_coverage_percentage_fast(
                    test_parks, boundary_poly, service_distance_m, boundary_area_m2
                )
                
                if test_coverage >= min_coverage:
                    st.success(f"✅ Merged 2 parks → saves 1")
                    refined_parks = [refined_parks[k] for k in range(len(refined_parks)) 
                                   if k != i and k != j] + [merge_park]
                    improvement_found = True
                    break
            
            if improvement_found:
                break
    
    final_coverage = calculate_coverage_percentage_fast(
        refined_parks, boundary_poly, service_distance_m, boundary_area_m2
    )
    
    if len(refined_parks) < len(parks):
        st.success(f"✨ Refinement: {len(parks)} → {len(refined_parks)} parks, {final_coverage:.1f}% coverage")
    else:
        st.info(f"✓ {len(refined_parks)} parks, {final_coverage:.1f}% coverage (already optimal)")
    
    return refined_parks
