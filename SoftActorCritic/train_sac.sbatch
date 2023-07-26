#!/bin/bash
####
#a) Define slurm job parameters
####
#SBATCH --job-name=SacTraining

#resources:
#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.
# (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one

#node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce
#1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=07:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours",
# "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err
#SBATCH --output=job.%J.out

# write the standard output to job.*jobID*.out
#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out
#SBATCH --mail-user=nion-immanuel.schuermeyer@student.uni-tuebingen.de
# your mail address


#c) Execute your code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still
# have them if your job fails
####
# source /home/${USER}/gym-rl/bin/activate
# python train_sac.py /scratch/$SLURM_JOB_ID/sac_training/
# deactivate

singularity exec --nv /home/"${USER}"/rl_lecture python3 /home/"${USER}"/sac_training/train_sac.py -e=25000

echo DONE!
