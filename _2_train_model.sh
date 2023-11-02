for MAP_SIZE in 6
do
# for STUDENT in 1 2 3 4 5 6 7 8 9 10
for STUDENT in 0
do
STUD1=$(($STUDENT+1))
STUD2=$(($STUDENT+2))
for TEACHER in $(seq $STUDENT 1 15);
# for TEACHER in 20
do
# for TRAIN in 64000
for TRAIN in 4096000
do
# for OFFSET in {0..19}
for OFFSET in {0..0}
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
    "cd ~/gymnasium-test && /opt/conda/bin/python _2_train_model.py" ${MAP_SIZE} ${STUDENT} ${TEACHER} ${TRAIN} 4 256 $OFFSET
    sleep 0.1
done
done
done
done
done
