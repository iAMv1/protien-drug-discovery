"""
        interaction_types = ['Hydrophobic', 'H-bond Donor', 'H-bond Acceptor', 
                           'Electrostatic', 'Van der Waals', 'π-π Stacking', 'Cation-π']
        
        counts = np.random.poisson(3, len(interaction_types))
        colors = np.random.uniform(0.3, 1.0, len(interaction_types))
        
        n_residues = min(20, len(protein_seq))
        residues = [f"Res{i+1}" for i in range(n_residues)]
        
        contribution_matrix = np.random.beta(1, 3, (len(interaction_types), n_residues))
        
        return {
            'interaction_types': {
                'types': interaction_types,
                'counts': counts.tolist(),
                'colors': colors.tolist()
            },
            'residue_contributions': {
                'matrix': contribution_matrix.tolist(),
                'residues': residues,
                'interaction_types': interaction_types
            }
        }