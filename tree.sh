# for SEED in 0
# for SEED in {100..900..100}
for SEED in {1000..4900..100}
do
SEED1=$((${SEED}+1))
SEED10=$((${SEED}+10))
SEED20=$((${SEED}+20))
SEED100=$((${SEED}+100))

# for BUDGET in 10 20 50 100 200 500 1000 2000 5000
# for BUDGET in 1 2 5 10 20 50 100 200 500 1000
for BUDGET in 1 2 4 8 16 32 64 128 256 512 1024 2048
# for BUDGET in 1 2 3 4 5 6 7 8 9 10
do

for DEPTH in 4
do
# 
for MODE in 'risk-0.5'
# for MODE in 'streak-2'
# for MODE in 'linear-1' 'linear-2' 'exponential-2'
do

for LR in 1
do

for OPT in 'adam'
do

echo "TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}_${LR}_${OPT}"

# cd ~/chesstrainer && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python 03_train_valid.py ${MODE} ${BUDGET} ${EPOCHS} ${PATIENCE} ${SEED} ${SEED4}

bsub -n 1 \
-q general \
-m general \
-G compute-chien-ju.ho \
-J TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}_${LR}_${OPT} \
-M 16GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREE_${SEED}_${BUDGET}_${DEPTH}_${MODE}_${LR}_${OPT}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=16GB] span[hosts=1]" \
-g /saumik/limit100 \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py" ${DEPTH} ${MODE} ${BUDGET} ${SEED}-${SEED100} ${OPT} ${LR} ${SKILL}

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