import os
from time import sleep

fieldpoint_types = ["negElec", "posElec", "vdw", "hydrophobic"]
# fieldpoint_types=["hydrophobic"]

fieldpoint_type = fieldpoint_types[3]

os.chdir(f"{fieldpoint_type}")

for ind in range(1, 199):
    print("sample number {ind} of {fieldpoint_type}")
    print("pymol cloud")
    os.system(f"pymol unclustered/cloud{ind}.pml")
    print("pymol cluster")
    os.system(f"pymol clustered/cluster{ind}.pml")
    sleep(1)
