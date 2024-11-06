# for SEED in 0
for SEED in {100..190..10}
do
SEED10=$((${SEED}+10))
SEED100=$((${SEED}+100))

for BUDGET_TYPE in equal
# for BUDGET_TYPE in equal left right
do

for BUDGET in 4 8 16 32 64 128 256 512 # 0-200
# for BUDGET in 1024 2048 4096 8192 16384 32768 65536 131072 # 0-100
do

# for DEPTH in 2 4
for DEPTH in 4
do


# for MODE in 'critroot-2-1-0.0' 'critroot-2-1-0.2' 'critroot-2-1-0.4' 
# for MODE in 'critroot-10-9-0.0' 'critroot-10-9-0.2' 'critroot-10-9-0.4' 
for MODE in 'crit-0-1-10-0.2' 'crit-0-1-10-0.4'
do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}"


# -q general \
# -m general \
# -g /saumik/limit100 \

bsub -n 1 \
-q general \
-m general \
-g /saumik/limit100 \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE} \
-M 8GB \
-N \
-u saumik@wustl.edu \
-o /scratch1/fs1/chien-ju.ho/Active/output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=8GB] span[hosts=1]" \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /scratch1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py single" ${DEPTH} ${MODE} ${BUDGET_TYPE}_${BUDGET} ${SEED}-${SEED10}

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