"""
PHYS432 W2022
Simulation of basaltic (mafic) lava flow down an inclined plane.
All quantities in SI units. Steady-state solution is plotted in black.
@author: Yuliya Shpunarska
@collab: Maya Tatarelli
Mar. 12th, 2022
"""
import numpy as np
import matplotlib.pyplot as pl
from numpy import random

## SETUP
# Set up the grid and diffusion parameters
Ngrid = 50
Nsteps = 5000
dt = 1
dx = 1

x = np.arange(0, Ngrid*1., dx) / Ngrid # multiplying by 1. to make sure this is an array of floats not integers

# Gravitational part: this is constant (working in SI units)
g = 9.81
alpha = 10 # inclination of slope in degrees
rho = 2700 # density of basaltic lava (from Wikipedia)
K = dt*g*np.sin(np.deg2rad(alpha))/rho

# viscosity of basaltic lava (from Wikipedia, kinematic viscosity = 10^4 cP)
v = 1e5/rho
beta = v*dt/dx**2

# Initial speed of the lava (pick that lava is initially at rest)
#f1 = random.uniform(low=0, high=0.005, size=x.shape)
f1 = np.zeros(x.shape)

## PLOTTING
# Set up plot
pl.ion()
fig, axes = pl.subplots(1,1)
axes.set_title("Flow of Basaltic Lava")
axes.set_ylabel("Speed of lava [m/s]")
axes.set_xlabel("Height of lava [m]")

# Steady state solution (as found in class)
H = 1 # height of lava layer
u_steady = -g*np.sin(np.deg2rad(alpha))/(2*v)*x**2 + g*np.sin(np.deg2rad(alpha))/v * H * x

# Plotting steady state in the background for reference
axes.plot(x, u_steady, 'k-')

# We will be updating these plotting objects
plt1, = axes.plot(x, f1, 'ro')

# Setting the axis limits for visualization
axes.set_xlim([0,H])
axes.set_ylim([0,0.03]) # From video seen in class we expect the speed to be ~1 cm/s

# this draws the objects on the plot
fig.canvas.draw()

for ct in range(Nsteps):

    ## DIFFUSION
    # Setting up matrices for diffusion operator
    A1 = np.eye(Ngrid) * (1.0 + 2.0 * beta) + np.eye(Ngrid, k=1) * -beta + np.eye(Ngrid, k=-1) * -beta

    ## BOUNDARY CONDITIONS
    # No-slip boundary condition on left side
    f1[0] = 0

    # Stress-free boundary condition on the right
    A1[Ngrid - 1][Ngrid - 1] = beta + 1.0

    # UPDATING
    f1 += K # add the gravitational component
    f1 = np.linalg.solve(A1, f1) # update with diffusion

    # update the plot
    plt1.set_ydata(f1)

    fig.canvas.draw()
    pl.pause(0.001)
