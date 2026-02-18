# DNA Helix

This project generates a 3D DNA double helix using Procedural Mesh Generation and Phong Lighting.

### Sample Video

https://github.com/user-attachments/assets/7670d264-a316-48b0-80b9-b294bf5ce2a2


## Control

- wasd to move
- space to pause rotation

## Logic

1. use spiral formula (Sine/Cosine) to create two "guide wires" in 3D space.
2. place a circle of vertices (a "ring") at every point along that path.
3. connect these rings using triangles to form solid tubes (the strands).
4. connect the two strands with small horizontal cylinders at regular intervals
5. to build light calculate "Normal" vectors for every point to make light bounces off the tubes, so it look round and 3D

## Logic Detail

### DNA Skeleton

use this equation

- x=R⋅cos(2π⋅turns⋅t+phase)
- y=height⋅(t−0.5)
- z=R⋅sin(2π⋅turns⋅t+phase)

phase from 0 to 180 degrees.

### Indexing

after create rings of vertices, use EBO to make them together

### Shading

- use Phong Reflection Model
- light depend on angle between the surface and source
