for MAP_SIZE in 12
do
for STUDENT in 3 5 7 9 11 13 15
do
# for TEACHER in {$STUDENT..20..2}
STUD1=$(($STUDENT+1))
for TEACHER in $(seq $STUD1 2 20);
# for MYOPIC in {1..30}
do
    echo $MAP_SIZE $STUDENT $TEACHER
    bsub -n 32 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${MAP_SIZE}_${TEACHER} \
    -M 64GB \
    -N \
    -u saumik@wustl.edu \
    -o /home/n.saumik/gymnasium-test/tmp/${MAP_SIZE}_${TEACHER}.%J \
    -R "rusage[mem=64GB] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=yes" \
    -g /saumik/limit10 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python 5_curriculums.py" ${MAP_SIZE} ${STUDENT} ${TEACHER} 100
    sleep 0.1
done
done
done