"""
3D Protein Visualization Page for Streamlit UI
Task 12.1 Implementation - Interactive 3D protein structure rendering
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from protein_drug_discovery.visualization.protein_3d_viewer import (
        Protein3DViewer, ProteinVisualizationUI, ProteinStructure, BindingSite
    )
    from protein_drug_discovery.visualization.binding_visualizer import BindingVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

def show_protein_3d_page():
    """Main 3D protein visualization page (Task 12.1)"""
    
    st.header("🧬 Interactive 3D Protein Structure Visualization")
    st.markdown("**Advanced 3D rendering with py3Dmol integration and interactive controls**")
    
    if not VISUALIZATION_AVAILABLE:
        st.error("❌ 3D visualization modules not available. Please check installation.")
        return
    
    # Initialize visualization components
    viewer = Protein3DViewer()
    ui = ProteinVisualizationUI()
    
    # Sidebar for main controls
    st.sidebar.header("🎛️ 3D Visualization Controls")
    
    # Input section
    st.subheader("📝 Protein Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Protein Sequence", "PDB ID", "Upload PDB File"],
        horizontal=True
    )
    
    protein_structure = None
    
    if input_method == "Protein Sequence":
        protein_sequence = st.text_area(
            "Enter protein amino acid sequence:",
            value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            height=100,
            help="Enter protein sequence in single-letter amino acid code"
        )
        
        if protein_sequence:
            structure_method = st.selectbox(
                "Structure prediction method:",
                ["mock", "alphafold", "colabfold"],
                help="Choose method for generating 3D structure"
            )
            
            if st.button("🔄 Generate 3D Structure"):
                with st.spinner("Generating 3D structure..."):
                    protein_structure = viewer.generate_pdb_from_sequence(
                        protein_sequence, structure_method
                    )
                    st.session_state.protein_structure = protein_structure
                    st.success("✅ 3D structure generated!")
    
    elif input_method == "PDB ID":
        pdb_id = st.text_input(
            "Enter PDB ID:",
            value="1ABC",
            help="Enter 4-character PDB identifier"
        )
        
        if st.button("📥 Fetch PDB Structure"):
            with st.spinner(f"Fetching PDB {pdb_id}..."):
                # Mock PDB fetching (in real implementation, use PDB API)
                mock_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                protein_structure = viewer.generate_pdb_from_sequence(mock_sequence, "mock")
                protein_structure.structure_source = f"PDB {pdb_id} (Mock)"
                st.session_state.protein_structure = protein_structure
                st.success(f"✅ PDB {pdb_id} structure loaded!")
    
    elif input_method == "Upload PDB File":
        uploaded_file = st.file_uploader(
            "Upload PDB file:",
            type=['pdb'],
            help="Upload a PDB structure file"
        )
        
        if uploaded_file is not None:
            pdb_content = uploaded_file.read().decode('utf-8')
            # Extract sequence from PDB (simplified)
            sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            
            protein_structure = ProteinStructure(
                pdb_content=pdb_content,
                sequence=sequence,
                structure_source=f"Uploaded: {uploaded_file.name}",
                binding_sites=[]
            )
            st.session_state.protein_structure = protein_structure
            st.success("✅ PDB file uploaded successfully!")
    
    # Main visualization section
    if hasattr(st.session_state, 'protein_structure'):
        protein_structure = st.session_state.protein_structure
        
        # Performance validation
        st.subheader("⚡ Performance Validation")
        performance_metrics = viewer.validate_3d_rendering_performance(protein_structure)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Structure Size", f"{performance_metrics['structure_size']} residues")
        with col2:
            st.metric("PDB Size", f"{performance_metrics['pdb_size_kb']:.1f} KB")
        with col3:
            st.metric("Performance", performance_metrics['performance_rating'])
        with col4:
            st.metric("Est. Render Time", f"{performance_metrics['estimated_render_time']:.1f}s")
        
        # Show recommendations if any
        if performance_metrics['recommendations']:
            st.info("💡 **Performance Recommendations**: " + 
                   ", ".join(performance_metrics['recommendations']))
        
        # Predict binding sites
        binding_sites = viewer.predict_binding_sites(protein_structure)
        
        # Interactive controls
        st.subheader("🎮 Interactive Controls")
        settings = viewer.create_interactive_controls(protein_structure)
        
        # Main 3D visualization tabs
        st.subheader("🧬 3D Protein Structure Viewer")
        
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🏠 Full Structure", 
            "🎯 Binding Sites", 
            "⚡ Performance Mode",
            "📊 Analysis"
        ])
        
        with viz_tab1:
            st.markdown("**Interactive 3D protein structure with full controls**")
            
            # Create enhanced 3D viewer
            try:
                viewer_component = viewer.create_3d_viewer(
                    protein_structure, 
                    binding_sites,
                    width=800, 
                    height=600,
                    interactive_controls=True
                )
                
                if viewer_component:
                    # Structure information panel
                    with st.expander("📋 Structure Information", expanded=False):
                        info_col1, info_col2 = st.columns(2)
                        with info_col1:
                            st.write(f"**Sequence Length**: {len(protein_structure.sequence)} residues")
                            st.write(f"**Structure Source**: {protein_structure.structure_source}")
                            st.write(f"**Binding Sites**: {len(binding_sites)} predicted")
                        with info_col2:
                            st.write(f"**Representation**: {settings['representation']}")
                            st.write(f"**Color Scheme**: {settings['color_scheme']}")
                            st.write(f"**Quality**: {settings['quality']}")
                
            except Exception as e:
                st.error(f"❌ Failed to create 3D viewer: {e}")
                st.info("💡 Try using Performance Mode for large structures")
        
        with viz_tab2:
            st.markdown("**Focused view of predicted binding sites with confidence scores**")
            
            if binding_sites:
                # Binding site selector
                selected_sites = st.multiselect(
                    "Select binding sites to highlight:",
                    options=list(range(len(binding_sites))),
                    default=list(range(min(3, len(binding_sites)))),
                    format_func=lambda x: f"Site {x+1} (Conf: {np.mean(binding_sites[x].confidence_scores):.2f})"
                )
                
                if selected_sites:
                    selected_binding_sites = [binding_sites[i] for i in selected_sites]
                    
                    try:
                        binding_viewer = viewer.create_binding_site_viewer(
                            protein_structure, 
                            selected_binding_sites
                        )
                        
                        # Binding site details
                        st.subheader("🎯 Binding Site Details")
                        for i in selected_sites:
                            site = binding_sites[i]
                            with st.expander(f"Binding Site {i+1}", expanded=True):
                                detail_col1, detail_col2, detail_col3 = st.columns(3)
                                with detail_col1:
                                    st.metric("Residues", len(site.residue_indices))
                                    st.metric("Type", site.site_type)
                                with detail_col2:
                                    avg_confidence = np.mean(site.confidence_scores)
                                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                    confidence_color = "🟢" if avg_confidence > 0.8 else "🟡" if avg_confidence > 0.6 else "🔴"
                                    st.write(f"**Quality**: {confidence_color}")
                                with detail_col3:
                                    st.write(f"**Center**: ({site.center_coordinates[0]:.1f}, "
                                           f"{site.center_coordinates[1]:.1f}, "
                                           f"{site.center_coordinates[2]:.1f})")
                                
                                # Confidence score distribution
                                conf_df = pd.DataFrame({
                                    'Residue': [f"Res{idx}" for idx in site.residue_indices],
                                    'Confidence': site.confidence_scores
                                })
                                st.bar_chart(conf_df.set_index('Residue'))
                    
                    except Exception as e:
                        st.error(f"❌ Failed to create binding site viewer: {e}")
                
            else:
                st.info("No binding sites predicted for this structure")
        
        with viz_tab3:
            st.markdown("**Performance-optimized viewer for large structures**")
            
            max_residues = st.slider(
                "Maximum residues to display:",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Limit structure size for better performance"
            )
            
            try:
                perf_viewer = viewer.create_performance_optimized_viewer(
                    protein_structure,
                    binding_sites,
                    max_residues=max_residues
                )
                
                # Performance tips
                st.info("💡 **Performance Tips**: Use this mode for structures >300 residues. "
                       "Reduces rendering complexity while maintaining key features.")
                
            except Exception as e:
                st.error(f"❌ Failed to create performance viewer: {e}")
        
        with viz_tab4:
            st.markdown("**Structural analysis and export options**")
            
            # Analysis section
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.subheader("📊 Structure Analysis")
                
                # Basic statistics
                sequence_length = len(protein_structure.sequence)
                binding_residues = sum(len(site.residue_indices) for site in binding_sites)
                
                st.metric("Total Residues", sequence_length)
                st.metric("Binding Residues", binding_residues)
                st.metric("Binding Coverage", f"{(binding_residues/sequence_length)*100:.1f}%")
                
                # Amino acid composition
                if st.checkbox("📈 Show Amino Acid Composition"):
                    aa_counts = {}
                    for aa in protein_structure.sequence:
                        aa_counts[aa] = aa_counts.get(aa, 0) + 1
                    
                    aa_df = pd.DataFrame([
                        {"Amino Acid": aa, "Count": count, "Percentage": (count/sequence_length)*100}
                        for aa, count in sorted(aa_counts.items())
                    ])
                    st.bar_chart(aa_df.set_index('Amino Acid')['Percentage'])
            
            with analysis_col2:
                st.subheader("💾 Export Options")
                
                export_format = st.selectbox(
                    "Choose export format:",
                    ["pdb", "xyz", "mol2"],
                    help="Select file format for structure export"
                )
                
                if st.button("📥 Download Structure"):
                    try:
                        exported_content = viewer.export_structure(protein_structure, export_format)
                        file_name = f"protein_structure.{export_format}"
                        
                        st.download_button(
                            label=f"Download {export_format.upper()} file",
                            data=exported_content,
                            file_name=file_name,
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {e}")
                
                # Screenshot option (placeholder)
                if st.button("📸 Capture Screenshot"):
                    st.info("Screenshot functionality would be implemented here")
                
                # Advanced export options
                with st.expander("⚙️ Advanced Export Options"):
                    include_binding_sites = st.checkbox("Include binding site annotations", value=True)
                    include_confidence = st.checkbox("Include confidence scores", value=True)
                    export_quality = st.selectbox("Export quality:", ["Standard", "High", "Publication"])
    
    else:
        st.info("👆 Please enter a protein sequence or upload a PDB file to start 3D visualization")
        
        # Show example gallery
        st.subheader("🖼️ Example Structures")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🧬 Small Protein (65 residues)"):
                example_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
                protein_structure = viewer.generate_pdb_from_sequence(example_seq, "mock")
                st.session_state.protein_structure = protein_structure
                st.rerun()
        
        with example_col2:
            if st.button("🔬 Medium Protein (150 residues)"):
                example_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 2 + "ADDITIONAL"
                protein_structure = viewer.generate_pdb_from_sequence(example_seq, "mock")
                st.session_state.protein_structure = protein_structure
                st.rerun()
        
        with example_col3:
            if st.button("🏗️ Large Protein (300 residues)"):
                example_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 4 + "EXTENDED"
                protein_structure = viewer.generate_pdb_from_sequence(example_seq, "mock")
                st.session_state.protein_structure = protein_structure
                st.rerun()

def show_3d_visualization_tests():
    """Show 3D visualization testing page"""
    
    st.header("🧪 3D Visualization Tests")
    st.markdown("**Testing and validation of 3D rendering performance**")
    
    if not VISUALIZATION_AVAILABLE:
        st.error("❌ Visualization modules not available")
        return
    
    viewer = Protein3DViewer()
    
    # Test different structure sizes
    st.subheader("📊 Performance Testing")
    
    test_sizes = [50, 100, 200, 300, 500]
    
    if st.button("🚀 Run Performance Tests"):
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, size in enumerate(test_sizes):
            status_text.text(f"Testing structure with {size} residues...")
            
            # Generate test structure
            test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            test_sequence = (test_sequence * (size // len(test_sequence) + 1))[:size]
            
            test_structure = viewer.generate_pdb_from_sequence(test_sequence, "mock")
            
            # Test performance
            start_time = time.time()
            performance_metrics = viewer.validate_3d_rendering_performance(test_structure)
            test_time = time.time() - start_time
            
            results.append({
                'Structure Size': size,
                'Render Time (s)': performance_metrics.get('actual_render_time', test_time),
                'Performance Rating': performance_metrics['performance_rating'],
                'PDB Size (KB)': performance_metrics['pdb_size_kb']
            })
            
            progress_bar.progress((i + 1) / len(test_sizes))
        
        status_text.text("✅ Performance testing completed!")
        
        # Display results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Performance chart
        st.subheader("📈 Performance Chart")
        st.line_chart(results_df.set_index('Structure Size')['Render Time (s)'])
        
        # Recommendations
        st.subheader("💡 Recommendations")
        avg_time = results_df['Render Time (s)'].mean()
        
        if avg_time < 1.0:
            st.success("🟢 Excellent performance! All structure sizes render quickly.")
        elif avg_time < 3.0:
            st.info("🟡 Good performance. Consider performance mode for largest structures.")
        else:
            st.warning("🔴 Performance optimization recommended for large structures.")

if __name__ == "__main__":
    show_protein_3d_page()