OPTIONS = {
    'HYDROGEN': {
        'geometry': lambda x: [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_H2_{x}"
    },
    'HYDROGEN_CHAIN': {
        'geometry': lambda x: [
                ('H', (0.0, 0.0, 0.0)),
                ('H', (0.0, 0.0, x)),
                ('H', (0.0, 0.0, 2*x)),
                ('H', (0.0, 0.0, 3*x))
            ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_chain_H4_{x}"
    },
    'LITHIUM_HYDRIDE': {
        'geometry': lambda x: [('Li', (0., 0., 0.)), ('H', (0., 0., x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Lithium_hydride_LiH_{x}"
    },
    'HYDROGEN_FLUORIDE': {
        'geometry': lambda x: [('H', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_fluoride_HF_{x}"
    },
    'WATER': {
        'geometry': lambda x: [ # triangular
            ('O', (0., 0., 0.)),
            ('H', (x, x, 0.)),
            ('H', (-x, x, 0.))
        ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Water_H2O_{x}"
    },
    'AMMONIA': {
        'geometry': lambda x, y, z: [
            ('N', (0., 0., 0.)),
            ('H', (0., 2*y, -z)),  # H1
            ('H', (-x, -y, -z)),   # H2
            ('H', (x, -y, -z))     # H3
        ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x, y, z: f"Ammonia_NH3_{x}_{y}_{z}"
    },
    'NITROGEN': {
        'geometry': lambda x: [('N', (0., 0., 0.)), ('N', (0., 0., x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Nitrogen_N2_{x}"
    }
}