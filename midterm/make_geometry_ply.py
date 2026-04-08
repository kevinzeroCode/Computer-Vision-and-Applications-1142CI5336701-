"""
make_geometry_ply.py
Combine Santa.xyz vertices + face data from SantaTriangle4Test.ply
to produce Santa_geometry.ply (a full mesh MeshLab can open).
Run once before picking points in MeshLab.
"""
import numpy as np

XYZ_FILE  = '7Images and xyz/Santa.xyz'
PLY_TMPL  = 'Supplementary/SantaTriangle4Test.ply'
OUT_PLY   = 'Santa_geometry.ply'

print('Loading Santa.xyz ...')
data = np.loadtxt(XYZ_FILE)
pts  = data[:, :3]
N    = len(pts)
print(f'  {N} vertices')

print('Reading PLY template for face data ...')
with open(PLY_TMPL, 'r') as f:
    lines = f.readlines()

# Find the INSERT marker line (where vertex data should go)
# and find the start of face data (lines starting with "3 ")
insert_idx = None
for i, l in enumerate(lines):
    if '[INSERT YOUR RESULT HERE' in l:
        insert_idx = i
        break

# Lines after insert marker = face data
face_lines = lines[insert_idx + 1:]

# Build new PLY header
header = [
    'ply\n',
    'format ascii 1.0\n',
    f'element vertex {N}\n',
    'property float x\n',
    'property float y\n',
    'property float z\n',
    'element face 190808\n',
    'property list uchar int vertex_indices\n',
    'end_header\n',
]

print(f'Writing {OUT_PLY} ...')
with open(OUT_PLY, 'w') as f:
    f.writelines(header)
    for x, y, z in pts:
        f.write(f'{x} {y} {z}\n')
    f.writelines(face_lines)

print(f'Done → {OUT_PLY}')
print('Open this file in MeshLab, then use Edit > Pick Points.')
