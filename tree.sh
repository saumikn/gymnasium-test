# for SEED in 0
for SEED in {0..90..10}
# for SEED in {200..490..10}
do
SEED10=$((${SEED}+10))
SEED100=$((${SEED}+100))

for BUDGET_TYPE in equal
# for BUDGET_TYPE in equal left right
do

# for BUDGET in 4 16 64 256 1024 4096 16384 65536
for BUDGET in 65536 262144 1048576
# for BUDGET in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
# for BUDGET in 8 32 128 512 2048 8192 32768 131072 524288
do

# for DEPTH in 2 4
for DEPTH in 4
do


# for MODE in 'normal-0-0' 'normal-1-0' 'normal-2-0'
# for MODE in 'normal-0-0' 'normal-2-0' 'normal-0-2'
# for MODE in 'uniform-50-150' 'uniform-0-200'
for MODE in 'critical-1-2'
do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}"


bsub -n 1 \
-q general \
-m general \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE} \
-M 64GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=64GB] span[hosts=1]" \
-g /saumik/limit100 \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py single" ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${SEED}-${SEED10}

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