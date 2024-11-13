# for SEED in {100..290..10}
for SEED in {0..900..100}
do
SEED10=$((${SEED}+10))
SEED100=$((${SEED}+100))
SEED500=$((${SEED}+500))

for DEPTH in 3 4 5
do

for FREQ in 0.2 0.1
do

# for MODE in 'crit-0-2-16' # 'crit-0-4-16' 'crit-0-1-8' # 'crit-0-1-16' 'crit-0-2-8' 
# for MODE in 'crit-1-2-0' 'crit-2-4-0' 'crit-4-6-0'
for MODE in 'crit-9-10-0' 'crit-8-10-0' 'crit-0-1-9' 'crit-0-2-8'
do

for BUDGET_TYPE in equal
# for BUDGET_TYPE in equal left right
do

# for BUDGET in 16 32 64 128 256 512 1024 2048 4096 # 8192 16384 32768 65536 131072 262144 524288 1048576 # 0-200
# for BUDGET in 8192 16384 32768 65536 131072 262144
# for BUDGET in 16 64 256 1024 4096 16384 65536
# for BUDGET in 32 128 512 2048 8192 32768 131072 
for BUDGET in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072
# for BUDGET in 262144
do

for STEPS in 32768 # 65536 131072
# for THRESH in 0 0.5 1
# for THRESH in 1
do

echo "TREE_${SEED}_${DEPTH}_${MODE}_${BUDGET}"

bsub -n 1 \
-q general \
-m general \
-g /saumik/limit100 \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE} \
-M 16GB \
-N \
-u saumik@wustl.edu \
-o /scratch1/fs1/chien-ju.ho/Active/output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-407.ris.wustl.edu'] rusage[mem=16GB] span[hosts=1]" \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /scratch1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree_discrim.py single" ${SEED}-${SEED100} ${DEPTH} ${MODE}-${FREQ} ${BUDGET_TYPE}_${BUDGET} ${STEPS}

# sleep 1

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

# break
done