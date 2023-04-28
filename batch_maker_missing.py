def slurm(dyad,iter,long_time=False):
    if long_time: 
        hour = 16
    else:
        hour = 1
    return f'''#!/bin/bash -l
#SBATCH -A snic2022-5-587
#SBATCH -n 2

#SBATCH -t {hour}:00:00
#SBATCH -J gpp_kgi{dyad}_{iter}
#SBATCH -o /proj/snic2019-3-404/logs_gpp/log_dyad{dyad}_{iter}.log

cd ~/gpp
conda activate /proj/snic2019-3-404/conda/gpp2
python -u run2.py --kgi=.2 --bias=.1 --iter={iter} --dyad={dyad}
'''
        
dyads = [750]
iters = [7, 8, 37, 38, 40, 42, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 63, 66, 68, 78, 81, 83, 86, 88, 89, 90, 92, 93, 94, 95, 96, 101, 113, 119, 120, 122, 132, 135, 136, 139, 140, 143, 146, 147, 148, 149, 150, 151, 153, 163, 171, 178, 233, 234, 235, 237, 241, 244, 246, 247, 248, 254, 256, 258, 259, 261, 262, 263, 264, 266, 268, 269, 270, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 285, 286, 287, 288, 291, 293, 295]
for iter in iters:
    for dyad in dyads:
        long = False
        if dyad in [640,746,750,865]:
            long = True
        files = slurm(dyad=dyad,iter=iter, long_time=long)
        print (files)
        with open(f"batcher/gpp_d{dyad}_i{iter}_k.2_b.1.sh", "w") as f:
            f.write(files)

