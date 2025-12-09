import argparse
from mpi4py import MPI
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import math

# MPI Init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    if rank == 0:
        print("Run with:   mpiexec --pernode -hostfile hostfilename python3 visualize.py --var temp")
    exit()

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--var", type=str, default="temp",
                    help="temp | sigma | fluor")
args = parser.parse_args()

choice = args.var.lower()
if choice not in ["temp", "sigma", "fluor"]:
    choice = "temp"

# Load CSV on Rank 0
if rank == 0:
    print("Loading predictions.csv...")
    df = pd.read_csv("predictions.csv")

    lat = df["Lat_Dec"].to_numpy()
    lon = df["Lon_Dec"].to_numpy()
    depth = df["Depth"].to_numpy()

    if choice == "temp":
        value = df["temp"].to_numpy()
        label = "Temperature (°C)"
    elif choice == "sigma":
        value = df["sigmatheta"].to_numpy()
        label = "Sigma-Theta (kg/m³)"
    else:
        value = df["fluor"].to_numpy()
        label = "Fluorescence (a.u.)"

    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    depth_min, depth_max = depth.min(), depth.max()
    val_min, val_max = value.min(), value.max()
else:
    lat = lon = depth = value = None
    lat_min = lat_max = lon_min = lon_max = 0
    depth_min = depth_max = val_min = val_max = 0
    label = None

# Broadcast
lat = comm.bcast(lat, root=0)
lon = comm.bcast(lon, root=0)
depth = comm.bcast(depth, root=0)
value = comm.bcast(value, root=0)

lat_min = comm.bcast(lat_min, root=0)
lat_max = comm.bcast(lat_max, root=0)
lon_min = comm.bcast(lon_min, root=0)
lon_max = comm.bcast(lon_max, root=0)
depth_min = comm.bcast(depth_min, root=0)
depth_max = comm.bcast(depth_max, root=0)
val_min = comm.bcast(val_min, root=0)
val_max = comm.bcast(val_max, root=0)
label   = comm.bcast(label,   root=0)

N = len(lat)

# Normalized Positions [-1,1]
def norm(v, vmin, vmax):
    """Normalize an array into the range [-1, 1] for OpenGL positioning."""
    return 2*((v - vmin)/(vmax - vmin + 1e-9)) - 1

X = norm(lon, lon_min, lon_max)
Y = norm(lat, lat_min, lat_max)
Z = norm(depth, depth_min, depth_max)
Z = -Z   # depth positive downward

Vn = (value - val_min)/(val_max - val_min + 1e-9)

# Colormap
def colormap(t):
    """
    Convert a normalized scalar t ∈ [0,1] into an RGB color using a fixed 
    perceptual gradient.
    """
    r = 0.267 + t*(0.993-0.267)
    g = 0.004 + t*(0.906-0.004)
    b = 0.329 + t*(0.143-0.329)
    return r,g,b

colors = np.array([colormap(v) for v in Vn])

# OPENGL + DISPLAY (FULLSCREEN)
os.environ["DISPLAY"] = ":0.0"
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Arial", 32)
font_small = pygame.font.SysFont("Arial", 20)

WIN_W, WIN_H = 1920, 1080
pygame.display.set_mode((WIN_W, WIN_H), OPENGL | DOUBLEBUF | FULLSCREEN)
pygame.mouse.set_visible(False)

glEnable(GL_DEPTH_TEST)
glEnable(GL_POINT_SMOOTH)
glPointSize(3.0)

# Alpha blending for text
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

glClearColor(0.02,0.02,0.03,1.0)

# 2×2 Tile Position
tile_x = rank % 2
tile_y = 1 - (rank // 2)

# OFF-AXIS Frustum
FOV = 45.0
aspect_global = 3840.0 / 2160.0
near, far = 0.1, 100.0

tan_half = math.tan(math.radians(FOV)*0.5)

global_top =  tan_half * near
global_bottom = -global_top
global_right = global_top * aspect_global
global_left  = -global_right

tl = tile_x * 0.5
tr = tl + 0.5
tb = tile_y * 0.5
tt = tb + 0.5

left   = global_left   + (global_right-global_left)*tl
right  = global_left   + (global_right-global_left)*tr
bottom = global_bottom + (global_top-global_bottom)*tb
top    = global_bottom + (global_top-global_bottom)*tt

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glFrustum(left, right, bottom, top, near, far)

# Draw Helpers
def draw_axes():
    """Render 3D reference axes (X,Y,Z) in the OpenGL scene."""
    glLineWidth(3)
    glBegin(GL_LINES)

    glColor3f(1,0,0) # X axis
    glVertex3f(-1,0,0); glVertex3f(1,0,0)

    glColor3f(0,1,0) # Y axis
    glVertex3f(0,-1,0); glVertex3f(0,1,0)

    glColor3f(0,0,1) # Z axis
    glVertex3f(0,0,-1); glVertex3f(0,0,1)

    glEnd()

def draw_ticks():
    """Draw small tick marks along each axis to give spatial scale reference."""
    glLineWidth(1)
    glColor3f(0.7,0.7,0.7)
    glBegin(GL_LINES)
    for i in range(-5,6):
        t = i * 0.2
        glVertex3f(t,0.01,0); glVertex3f(t,-0.01,0)
        glVertex3f(0.01,t,0); glVertex3f(-0.01,t,0)
        glVertex3f(0,0.01,t); glVertex3f(0,-0.01,t)
    glEnd()

def draw_bounding_box():
    """
    Draw a wireframe cube representing the normalized spatial bounds of 
    latitude, longitude, and depth.
    """
    glColor3f(0.5,0.5,0.5)
    glLineWidth(2)
    glBegin(GL_LINES)

    c = [
        (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
        (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)
    ]
    e = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
    for a,b in e:
        glVertex3f(*c[a]); glVertex3f(*c[b])
    glEnd()

def draw_origin():
    """Render a point at the coordinate origin for reference."""
    glPointSize(10)
    glBegin(GL_POINTS)
    glColor3f(1,1,1)
    glVertex3f(0,0,0)
    glEnd()
    glPointSize(3)

def draw_colorbar(screen_w, screen_h):
    """
    Draw a vertical RGB colorbar on the screen to represent scalar variable 
    magnitude, using the same colormap as the point cloud.
    """
    x0 = screen_w - 80
    y0 = int(screen_h*0.20)
    bar_h = int(screen_h*0.60 )
    bar_w = 20

    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0,screen_w,screen_h,0,-1,1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()

    glBegin(GL_QUADS)
    for i in range(bar_h):
        t = i/(bar_h-1)
        r,g,b = colormap(t)
        y = y0 + bar_h - i
        glColor3f(r,g,b)
        glVertex2f(x0, y)
        glVertex2f(x0+bar_w, y)
        glVertex2f(x0+bar_w, y-1)
        glVertex2f(x0, y-1)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

def draw_text(screen_w, screen_h, text, x, y, font_obj=None):
    """
    Render 2D text overlay on the screen using a Pygame font texture.

    Args:
        screen_w (int): Window width in pixels.
        screen_h (int): Window height in pixels.
        text (str): Text string to draw.
        x (int): Horizontal pixel position.
        y (int): Vertical pixel position.
        font_obj: Optional Pygame font object.
    """
    if font_obj is None:
        font_obj = font

    surf = font_obj.render(text, True, (255,255,255))
    raw = pygame.image.tostring(surf, "RGBA", True)
    w,h = surf.get_size()

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,raw)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)

    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0,screen_w,screen_h,0,-1,1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glColor4f(1,1,1,1)

    glBegin(GL_QUADS)
    glTexCoord2f(0,1); glVertex2f(x,   y)
    glTexCoord2f(1,1); glVertex2f(x+w, y)
    glTexCoord2f(1,0); glVertex2f(x+w, y+h)
    glTexCoord2f(0,0); glVertex2f(x,   y+h)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    glDeleteTextures([tex])

# Animation Camera
t = 0.0
radius = 4.0
cam_height = 0.5

clock = pygame.time.Clock()

# Main Loop
running = True
while running:

    if rank == 0:
        dt = clock.get_time()/1000.0
        t += dt*0.3

    t = comm.bcast(t, root=0)

    cam_x = radius*math.cos(t)
    cam_z = radius*math.sin(t)
    cam_y = cam_height

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(cam_x,cam_y,cam_z, 0,0,0, 0,1,0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_bounding_box()
    draw_axes()
    draw_ticks()
    draw_origin()

    glBegin(GL_POINTS)
    for i in range(N):
        glColor3f(colors[i,0], colors[i,1], colors[i,2])
        glVertex3f(X[i],Y[i],Z[i])
    glEnd()

    # Labels only on RANK 0
    if rank == 0:
        draw_text(WIN_W,WIN_H,f"{label}",20,20)
        draw_text(WIN_W,WIN_H,
                  f"Lat: {lat_min:.2f}→{lat_max:.2f}   Lon: {lon_min:.2f}→{lon_max:.2f}",
                  20,70)
        draw_text(WIN_W,WIN_H,
                  f"Depth: {depth_min:.1f}→{depth_max:.1f} m",
                  20,120)
        draw_text(WIN_W,WIN_H,
                  f"Value range: {val_min:.2f} → {val_max:.2f}",
                  20,170)
        draw_text(WIN_W, WIN_H,
                  f"Color scale: Low (purple) → High (yellow)",
                  20, 220)
        draw_text(WIN_W, WIN_H,
                "Axes: X = Longitude, Y = Latitude, Z = Depth",
                20, 270)


    # Colorbar only on RANK 1
    if rank == 1:
        draw_colorbar(WIN_W, WIN_H)
        draw_text(WIN_W, WIN_H, "Colorbar", WIN_W - 200, int(WIN_H*0.20), font_small)

    pygame.display.flip()
    clock.tick(60)
