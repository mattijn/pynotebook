# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

# <codecell>

import sqlite3

# <codecell>

import geojson

# <codecell>

# create a new Spatialite database file (and delete any existing of the same name)
if os.path.exists('MyDatabase.sqlite'):
    os.remove('MyDatabase.sqlite')
conn = sqlite3.connect('MyDatabase.sqlite')

# load spatialite extensions for SQLite.
# on Windows: make sure to have the mod_spatialite.dll somewhere in your system path.
# on Linux: to my knowledge the file is called mod_spatialite.so and also should be located in a directory available in your system path.
conn.enable_load_extension(True)
conn.execute('SELECT load_extension("mod_spatialite.dll")')

# <codecell>

conn

# <codecell>


