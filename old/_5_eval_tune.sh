for OFFSET in 0
do
for MAP_SIZE in 12
do
# for STUDENT in 1 3 5 7 9 11 13 15 17 19 21
for STUDENT in 2 4 6 8 10 12 14 16 18 20 21 22 23 24 25 26 27 28 29 30
do
# for TEACHER in {$STUDENT..20..2}
STUD1=$(($STUDENT+1))
STUD2=$(($STUDENT+2))
for TEACHER in $(seq $STUD1 1 $STUD1);
do
for TRAIN in 2048000
# for TRAIN in 2000 4000 8000 16000 32000 64000 128000 256000 512000 1024000 2048000
do
    echo $MAP_SIZE $STUDENT $TEACHER $TRAIN $OFFSET
    bsub -n 5 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${MAP_SIZE}_${STUDENT}_${TEACHER}_${TRAIN}_${OFFSET} \
    -M 16GB \
    -N \
    -u saumik@wustl.edu \
    -o /home/n.saumik/gymnasium-test/tmp/${MAP_SIZE}_${STUDENT}_${TEACHER}_${TRAIN}_${OFFSET}.%J \
    -R "rusage[mem=16GB] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=no" \
    -g /saumik/limit100 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python 5_eval_tune.py" ${MAP_SIZE} ${STUDENT} ${TEACHER} ${TRAIN} ${OFFSET}
    sleep 0.1
done
done
done
done
done