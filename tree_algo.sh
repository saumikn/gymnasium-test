# for SEED in 0
for SEED in {1..299}
# for SEED in {100..499}
do
SEED1=$((${SEED}+1))


for DEPTH in 4
do


# for MODE in 'crit-0-1-10-0.2'
for MODE in 'crit-8-10-0-0.2'
do

echo "TREEALGO_${SEED}_${DEPTH}_${MODE}"


bsub -n 1 \
-q general \
-m general \
-G compute-chien-ju.ho \
-J TREEALGO_${SEED}_${DEPTH}_${MODE} \
-M 16GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREEALGO_${SEED}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=16GB] span[hosts=1]" \
-g /saumik/limit100 \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py algo" ${DEPTH} ${MODE} 16 262144 1/8 2 ${SEED}-${SEED1}

# break
done

# break
done

# break
done