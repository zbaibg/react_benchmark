#!/bin/bash

mkdir -p xyz_files

for atom in Zn H C N O; do
    cat > xyz_files/${atom}_monomer.xyz <<EOF
1
$atom
$atom  0.0  0.0  0.0
EOF
done