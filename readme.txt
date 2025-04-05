To run the small simulation:
1) Go into params.py
2) Change line 15 to: nbody_file_in = "/home/jamesdr/scratch/smallbox/output_00011"
3) Change line 20 to: path = re.match(r'(.*/smallbox/)', nbody_file_in).group(1)
4) Change line 74 to: nbody_file_in = "/home/jamesdr/scratch/smallbox/output_00011"
5) Change line 79 to: path = re.match(r'(.*/smallbox/)', nbody_file_in).group(1)
6) Change line 91 to: "size": 'small',
7) Change line 108 to: nbody_file_in = "/home/jamesdr/scratch/smallbox/output_00011"
8) Change line 112 to: path = re.match(r'(.*/smallbox/)', nbody_file_in).group(1)
To run the large simulations:
1) Go into params.py
2) Change line 15 to: nbody_file_in = "/scratch/gpfs/rt3504/bigbox/output_00011"
3) Change line 20 to: path = re.match(r'(.*/bigbox/)', nbody_file_in).group(1)
4) Change line 74 to: nbody_file_in = "/scratch/gpfs/rt3504/bigbox/output_00011"
5) Change line 79 to: path = re.match(r'(.*/bigbox/)', nbody_file_in).group(1)
6) Change line 91 to: "size": 'big',
7) Change line 108 to: nbody_file_in = "/scratch/gpfs/rt3504/bigbox/output_00011"
8) Change line 112 to: path = re.match(r'(.*/bigbox/)', nbody_file_in).group(1)
