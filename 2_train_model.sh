for MAP_SIZE in 4
do
for STUDENT in 0
do
# for TEACHER in {$STUDENT..20..2}
STUD1=$(($STUDENT+1))
STUD2=$(($STUDENT+2))
for TEACHER in $(seq $STUD1 1 15);
# for TEACHER in 14 15 16 17 18 19 20
do
for TRAIN in 2048000
do
for OFFSET in 0
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
    "cd ~/gymnasium-test && /opt/conda/bin/python 2_train_model.py" ${MAP_SIZE} ${STUDENT} ${TEACHER} ${TRAIN} ${OFFSET}
    sleep 0.1
done
done
done
done
done
