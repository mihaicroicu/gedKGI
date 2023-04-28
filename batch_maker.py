def slurm(iter,long_time=False, k='.2', b='.1'):
    if long_time: 
        hour = 24
    else:
        hour = 1
    return f'''#!/bin/bash -l
#SBATCH -A snic2022-5-587
#SBATCH -n 1

#SBATCH -t {hour}:00:00
#SBATCH -J gpp_kgi_{iter}
#SBATCH -o /proj/snic2019-3-404/logs_gpp/log_{iter}.log

cd ~/gpp
conda activate /proj/snic2019-3-404/conda/gpp2
python -u run2.py --kgi={k} --bias={b} --iter={iter}
'''
        
for k in ['.1','.2','.3']:
 for b in ['0','.1','.2','.3']:
  for iter in range(100,300):
        files = slurm(iter=iter, long_time=True, k=k, b=b)
        print (files)
        with open(f"batcher/gpp_i{iter}_k{k}_b{b}.sh", "w") as f:
            f.write(files)

