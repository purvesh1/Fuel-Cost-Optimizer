# routes.py

# Coordinates for cities
cities_coords = {
    'Fresno, CA': (-119.7871, 36.7378),
    'Cheshire, CT': (-72.9106, 41.5084),
    'Maryland': (-76.6413, 39.0458),
    'Chicago, IL': (-87.6298, 41.8781),
    'Sacramento, CA': (-121.4944, 38.5816),
    'Los Angeles, CA': (-118.2437, 34.0522),
    'Atlanta, GA': (-84.3880, 33.7490),
    'Tulsa, OK': (-95.9928, 36.1539)
}

# Routes dictionary with associated default file paths
routes = {
    'Fresno to Cheshire via Tulsa': {
        'cities': ['Fresno, CA', 'Tulsa, OK', 'Cheshire, CT'],
        'default_file': 'fresno_to_cheshire_stations.xlsx'
    },
    'Cheshire to Fresno via Tulsa': {
        'cities': ['Cheshire, CT', 'Tulsa, OK', 'Fresno, CA'],
        'default_file': 'cheshire_to_fresno_stations.xlsx'
    },
    'Fresno to Maryland': {
        'cities': ['Fresno, CA', 'Maryland'],
        'default_file': 'fresno_to_maryland_stations.xlsx'
    },
    'Maryland to Fresno': {
        'cities': ['Maryland', 'Fresno, CA'],
        'default_file': 'maryland_to_fresno_stations.xlsx'
    },
    'Fresno to Chicago': {
        'cities': ['Fresno, CA', 'Chicago, IL'],
        'default_file': 'fresno_to_chicago_stations.xlsx'
    },
    'Chicago to Fresno': {
        'cities': ['Chicago, IL', 'Fresno, CA'],
        'default_file': 'chicago_to_fresno_stations.xlsx'
    },
    'Sacramento to Chicago': {
        'cities': ['Sacramento, CA', 'Chicago, IL'],
        'default_file': 'sacramento_to_chicago_stations.xlsx'
    },
    'Chicago to Sacramento': {
        'cities': ['Chicago, IL', 'Sacramento, CA'],
        'default_file': 'chicago_to_sacramento_stations.xlsx'
    },
    'LA to Chicago': {
        'cities': ['Los Angeles, CA', 'Chicago, IL'],
        'default_file': 'la_to_chicago_stations.xlsx'
    },
    'Chicago to LA': {
        'cities': ['Chicago, IL', 'Los Angeles, CA'],
        'default_file': 'chicago_to_la_stations.xlsx'
    },
    'Fresno to Atlanta': {
        'cities': ['Fresno, CA', 'Atlanta, GA'],
        'default_file': 'fresno_to_atlanta_stations.xlsx'
    },
    'Atlanta to Fresno': {
        'cities': ['Atlanta, GA', 'Fresno, CA'],
        'default_file': 'atlanta_to_fresno_stations.xlsx'
    }
}