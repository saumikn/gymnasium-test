# for SEED in 2000 2100
# for SEED in {2000..2090..10}
# for SEED in {10..90..10}
# for SEED in {0..90..10}
for SEED in 0
do
SEED10=$((${SEED}+10))
SEED100=$((${SEED}+100))
SEED500=$((${SEED}+500))

for DEPTH in 4
do

for FREQ in 0.1 0.2
do

# for MODE in 'crit-0-1-10-0.2' 'crit-0-1-10-0.4'
for MODE in 'crit-0-2-16' 'crit-0-4-16' 'crit-0-1-8' # 'crit-0-1-16' 'crit-0-2-8' 
# for MODE in 'crit-0-2-16'
# for MODE in 'crit-0-2-10-0.2' 'crit-0-4-10-0.2'
do

for BUDGET_TYPE in equal
# for BUDGET_TYPE in equal left right
do

# for BUDGET in 16 32 64 128 256 512 1024 2048 4096 # 8192 16384 32768 65536 131072 262144 524288 1048576 # 0-200
# for BUDGET in 8192 16384 32768 65536 131072 262144
for BUDGET in 16 64 256 1024 4096 16384 65536
# for BUDGET in 16384 65536 262144
do

for THRESH in 0 0.1 0.5 0.9 1
# for THRESH in 0 0.5 1
# for THRESH in 1
do

# for STEPS in 4096 # 16384 65536
for STEPS in 16384
# for THRESH in 0 0.5 1
# for THRESH in 1
do

echo "TREE_${SEED}_${DEPTH}_${MODE}_${BUDGET}_${THRESH}"


# -q general \
# -m general \
# -g /saumik/limit100 \

bsub -n 1 \
-q general \
-m general \
-g /saumik/limit100 \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}_${THRESH} \
-M 16GB \
-N \
-u saumik@wustl.edu \
-o /scratch1/fs1/chien-ju.ho/Active/output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}_${THRESH}.%J.txt \
-R "select[hname!='compute1-exec-407.ris.wustl.edu'] rusage[mem=16GB] span[hosts=1]" \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /scratch1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree_discrim.py single" ${SEED}-${SEED100} ${DEPTH} ${MODE}-${FREQ} ${BUDGET_TYPE}_${BUDGET} ${THRESH} ${STEPS}

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

# break
done