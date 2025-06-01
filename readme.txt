Hello, and welcome to the baryonification pipeline for use by miniRamses!

First, I assume you have downloaded the baryonification pipeline from Aurel Schneider. If not, you can find it here: https://bitbucket.org/aurelschneider/baryonification/src/develop/

Second, I assume you already have some miniramses simulation(s) you want to baryonify!

From here, we'll follow a step-by-step process to running your first baryonification simulation!

Step 1: Replacing io.py and params.pu

To begin, you'll want to alter some of the baryonification codes to let them read in miniramses simulations.

The crux of this is in the io.py and params.py files you can find in baryonification/baryonification. You can simply copy the io.py and params.py from this repository, and paste them over io.py and params.py

Step 2: Change params.py to read in your simulation

Before you can run the baryonification pipeline, you'll need to go into params.py and change a few things.

To do this (Note that things in ALL CAPS are things you should alter to suit your needs
1) Go into params.py
2) Change line 15 to: nbody_file_in = INSERT THE FILEPATH OF YOUR SIMULATION HERE
3) Change line 20 to: path = re.match(r'(.*/INSERT THE PART OF YOUR SIMULATION FILEPATH THAT COMES RIGHT BEFORE OUTPUT/)', nbody_file_in).group(1)
4) Change line 74 to: nbody_file_in = INSERT THE FILEPATH OF YOUR SIMULATION HERE
5) Change line 79 to: path = re.match(r'(.*/INSERT THE PART OF YOUR SIMULATION FILEPATH THAT COMES RIGHT BEFORE OUTPUT/)', nbody_file_in).group(1)
6) Change line 91 to: "size": 'large' or "size": 'small', depending on the size of your simulation (small = 100 Mpc/h, large = 200 Mpc/h)
7) Change line 108 to: nbody_file_in = INSERT THE FILEPATH OF YOUR SIMULATION HERE
8) Change line 112 to: path = re.match(r'(.*/INSERT THE PART OF YOUR SIMULATION FILEPATH THAT COMES RIGHT BEFORE OUTPUT/)', nbody_file_in).group(1)
9) Save it and you're done!

Step 2: Running the baryonification code.

With io.py and params.py fixed, we can now run the baryonification code. 
