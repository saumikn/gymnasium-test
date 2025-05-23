for SEED in {0..99}
# for SEED in {10..990..10}
# for SEED in {2000..3900..10}
do
SEED10=$((${SEED}+1))
# SEED10=$((${SEED}+10))
# SEED20=$((${SEED}+20))
# SEED100=$((${SEED}+100))

# for BUDGET_TYPE in equal
# for BUDGET_TYPE in 0 25 50 75 100
for BUDGET_TYPE in equal
do

# for BUDGET in 4 8 16 32 64 128 256 512 # 0-200
# for BUDGET in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072
# for BUDGET in 1048576
# for BUDGET in 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 # 0-100
# for BUDGET in 64 256
for BUDGET in 256 1024 4096 16384 65536 262144 # 0-100
do

for BATCH_SIZE in 32 # 4 16 64
do


for DEPTH in 4
do


# for MODE in 'binary' 'random' 'linear-2' 'linear-3' 'linear-4' 'exponential-2' 'exponential-3' 'exponential-4' 'crit-8-10-0-0.2'  'crit-0-2-8-0.2' 'crit-8-10-0-0.4' 'crit-8-10-0-0.6' 'crit-8-10-0-0.8' 'crit-8-10-0-1.0'
for MODE in 'random' 'crit-8-10-0-0.2' #'crit-0-2-8-0.2'

do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}"

mem=12


bsub -n 1 \
-q general \
-m general \
-g /saumik/limit300 \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE} \
-M ${mem}GB \
-N \
-u saumik@wustl.edu \
-o /scratch1/fs1/chien-ju.ho/Active/output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=${mem}GB] span[hosts=1]" \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py single" ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${BATCH_SIZE} ${SEED}-${SEED10}

# "cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py single" ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${SEED}-${SEED10}

# break
done

# break
done

# break
done

# break
done

# break
done

# break
done

# for SEED in 0; do SEED1=$((${SEED}+1)); SEED10=$((${SEED}+10)); for BUDGET_TYPE in equal left right; do for BUDGET in 256 1024 4096 16384 65536; do for DEPTH in 4; do MODE='crit-8-10-0-0.2'; time python tree.py single ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${SEED}-${SEED1}; done; done; done; done
