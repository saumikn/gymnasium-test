# for SEED in 0
for SEED in {0..199}
# for SEED in {200..490..10}
do
SEED1=$((${SEED}+1))


for DEPTH in 4
do


for MODE in 'critical-1-2'
do

echo "TREEALGO_${SEED}_${DEPTH}_${MODE}"


bsub -n 1 \
-q general \
-m general \
-G compute-chien-ju.ho \
-J TREEALGO_${SEED}_${DEPTH}_${MODE} \
-M 4GB \
-N \
-u saumik@wustl.edu \
-o /storage1/fs1/chien-ju.ho/Active/quickstart/job_output/TREEALGO_${SEED}_${DEPTH}_${MODE}.%J.txt \
-R "select[hname!='compute1-exec-401.ris.wustl.edu'] rusage[mem=4GB] span[hosts=1]" \
-g /saumik/limit100 \
-a "docker(saumikn/quickstart)" \
"cd ~/gymnasium-test && /storage1/fs1/chien-ju.ho/Active/.conda/envs/random/bin/python tree.py algo" ${DEPTH} ${MODE} 16 1000000 1/16 1.5 ${SEED}-${SEED1}

# break
done

# break
done

# break
done