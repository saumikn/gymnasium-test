for SEED in {0..190..10}
do
SEED10=$((${SEED}+10))

# for BUDGET in 10 20 50 100 200 500 1000 2000 5000
for BUDGET in 1 2 5
do

for DEPTH in 4 8
do

for SKILL in 2 4 6 8
do

for MODE in 'float'
do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${SKILL}_${MODE}"

# cd ~/chesstrainer && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python 03_train_valid.py ${MODE} ${BUDGET} ${EPOCHS} ${PATIENCE} ${SEED} ${SEED4}

bsub -n 1 \
-q general \
-m general \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${SKILL}_${MODE} \
-M 16GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREE_${SEED}_${BUDGET}_${DEPTH}_${SKILL}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=16GB] span[hosts=1]" \
-g /saumik/limit100 \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py" ${DEPTH} ${SKILL} ${MODE} ${BUDGET} ${SEED}-${SEED10}

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