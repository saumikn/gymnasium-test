for SEED in {100..190..10}
# for SEED in {100..190..10}
# for SEED in {200..490..10}
do
SEED10=$((${SEED}+10))
SEED100=$((${SEED}+100))

for BUDGET_TYPE in equal
# for BUDGET_TYPE in equal left right
do

for BUDGET in 16 64 256 # 1024 4096 16384 # 65536
# for BUDGET in 65536
# for BUDGET in 65536 262144 1048576
# for BUDGET in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
# for BUDGET in 8 32 128 512 2048 8192 32768 131072 524288
do

# for DEPTH in 2 4
for DEPTH in 4
do


# for MODE in 'critroot-2-1-0.0' 'critroot-2-1-0.2' 'critroot-2-1-0.4' 
# for MODE in 'critroot-10-9-0.0' 'critroot-10-9-0.2' 'critroot-10-9-0.4' 
# for MODE in 'crit123-9-10-0-0.2' 'crit123-0-1-9-0.2' 'crit123-9-10-0-0.4' 'crit123-0-1-9-0.4'
for MODE in 'crit123-0-1-10-0.2'
do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}"


# -q general \
# -m general \
# -g /saumik/limit100 \

bsub -n 1 \
-q general-interactive \
-m general-interactive \
-g /saumik/limit10 \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE} \
-M 8GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=8GB] span[hosts=1]" \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py single" ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${SEED}-${SEED10}

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