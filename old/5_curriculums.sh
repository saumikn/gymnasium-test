for MAP_SIZE in 12
do
for STUDENT in 5
do
# for TEACHER in {$STUDENT..20..2}
STUD1=$(($STUDENT+1))
STUD2=$(($STUDENT+2))
for TEACHER in $(seq $STUD1 1 30);
do
for TRAIN in 30000
do
for OFFSET in 0
do
    echo $MAP_SIZE $STUDENT $TEACHER $TRAIN $OFFSET
    bsub -n 10 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${MAP_SIZE}_${STUDENT}_${TEACHER}_${TRAIN}_${OFFSET} \
    -M 64GB \
    -N \
    -u saumik@wustl.edu \
    -o /home/n.saumik/gymnasium-test/tmp/${MAP_SIZE}_${STUDENT}_${TEACHER}_${TRAIN}_${OFFSET}.%J \
    -R "rusage[mem=64GB] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=yes" \
    -g /saumik/limit10 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python 5_curriculums.py" ${MAP_SIZE} ${STUDENT} ${TEACHER} ${TRAIN} ${OFFSET}
    sleep 0.1
done
done
done
done
done