#!/usr/bin/env python3
"""Odwraca normalne mesha OBJ przez:
  - negację wszystkich vn x y z → -x -y -z
  - odwrócenie winding order każdej ściany: f a b c → f a c b
Działa strumieniowo — nie ładuje całego pliku do pamięci.
"""
import sys
import os

def flip(src: str, dst: str) -> None:
    total_vn = 0
    total_f  = 0
    with open(src, "r") as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("vn "):
                parts = line.split()
                fout.write(f"vn {-float(parts[1])} {-float(parts[2])} {-float(parts[3])}\n")
                total_vn += 1
            elif line.startswith("f "):
                verts = line.split()[1:]   # ['a//na', 'b//nb', 'c//nc']
                # odwrócenie: a c b
                fout.write(f"f {verts[0]} {verts[2]} {verts[1]}\n")
                total_f += 1
            else:
                fout.write(line)
    print(f"Gotowe: {total_vn} normalnych, {total_f} ścian odwróconych → {dst}")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "tunel.obj"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".obj", "_flipped.obj")
    if not os.path.exists(src):
        print(f"Brak pliku: {src}")
        sys.exit(1)
    flip(src, dst)
