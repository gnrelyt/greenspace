
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
    
    # Only render if boundary exists
    if boundary is not None:
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
            
            # Helper function to find which parks are new/moved
            def get_changed_parks(current_step_data, previous_step_data=None):
                """Compare parks between steps to find which ones changed."""
                if previous_step_data is None:
                    # First step or no previous - highlight all
                    return current_step_data['parks']
                
                current_parks = current_step_data['parks']
                previous_parks = previous_step_data['parks']
                
                changed = []
                for cp in current_parks:
                    is_new = True
                    for pp in previous_parks:
                        # Check if parks are at the same location (within small tolerance)
                        if abs(cp.centroid.x - pp.centroid.x) < 0.00001 and \
                           abs(cp.centroid.y - pp.centroid.y) < 0.00001:
                            is_new = False
                            break
                    if is_new:
                        changed.append(cp)
                
                return changed
            
            if step_data['type'] == 'candidates':
                m = build_live_map(boundary, candidate_parks=step_data['parks'], bounds=bounds)
                st.markdown("### 🔵 Candidate Park Locations (Orange Grid)")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'demand_points':
                m = build_live_map(boundary, candidate_parks=step_data['parks'], 
                                 demand_points=step_data.get('demand_points'), bounds=bounds)
                st.markdown("### 🔵 Candidates (Orange) + 🔷 Demand Points (Blue)")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'optimal_solution':
                m = build_live_map(boundary, selected_parks=step_data['parks'],
                                 service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                                 bounds=bounds)
                st.markdown("### 🟢 ILP Optimal Solution")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'refinement':
                # Find which parks were merged (compare to previous step)
                previous_step = st.session_state.algorithm_steps[st.session_state.current_step - 1] if st.session_state.current_step > 0 else None
                highlight = get_changed_parks(step_data, previous_step)
                
                m = build_live_map(boundary, selected_parks=step_data['parks'],
                                 service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                                 bounds=bounds,
                                 highlight_parks=highlight)
                st.markdown(f"### 🟢 Parks (Green) | 🟡 Merged Parks (Yellow) | {step_data['description']}")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'position_optimization':
                # Find which parks were repositioned (compare to previous step)
                previous_step = st.session_state.algorithm_steps[st.session_state.current_step - 1] if st.session_state.current_step > 0 else None
                highlight = get_changed_parks(step_data, previous_step)
                
                m = build_live_map(boundary, selected_parks=step_data['parks'],
                                 service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                                 bounds=bounds,
                                 highlight_parks=highlight)
                st.markdown(f"### 🟢 Parks (Green) | 🟡 Repositioned Parks (Yellow) | {step_data['description']}")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'coverage_fine_tune':
                # Find which parks were fine-tuned (compare to previous step)
                previous_step = st.session_state.algorithm_steps[st.session_state.current_step - 1] if st.session_state.current_step > 0 else None
                highlight = get_changed_parks(step_data, previous_step)
                
                m = build_live_map(boundary, selected_parks=step_data['parks'],
                                 service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                                 bounds=bounds,
                                 highlight_parks=highlight)
                st.markdown(f"### 🟢 Parks (Green) | 🟡 Fine-tuned Parks (Yellow) | {step_data['description']}")
                st_folium(m, width=1400, height=600)
            elif step_data['type'] == 'final':
                m = build_live_map(boundary, selected_parks=step_data['parks'],
                                 service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                                 bounds=bounds)
                st.markdown(f"### ✅ Final Result | {step_data['description']}")
                st_folium(m, width=1400, height=600)
        else:
            # Fallback: show final result if no valid step
            m = build_live_map(
                boundary,
                selected_parks=st.session_state.parks,
                service_distance_m=calculate_service_distance(st.session_state.park_size_ha),
                bounds=bounds
            )
            st_folium(m, width=1400, height=600)
    else:
        st.warning("Could not load boundary for visualization")
        
st.divider()

st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    <p>🗺️ Green Mapping Tool | Optimal Park Placement with ILP + Continuous Refinement</p>
</div>
""", unsafe_allow_html=True)
