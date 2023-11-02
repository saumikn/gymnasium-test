for NODE in 256 1024
# for MAP_SIZE in 12
do
for STUDENT in 1 2 3 4 5 6 7 8 9 10
# for MYOPIC in {1..30}
do
STUD1=$(($STUDENT+1))
STUD2=$(($STUDENT+2))
for TEACHER in $(seq $STUDENT 1 15);
# for TEACHER in 20
do
    echo ${NODE}_${STUDENT}_${TEACHER}
    bsub -n 10 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${NODE}_${STUDENT}_${TEACHER} \
    -M 64GB \
    -N \
    -u saumik@wustl.edu \
    -o /home/n.saumik/gymnasium-test/tmp/${NODE}_${STUDENT}_${TEACHER}.%J \
    -R "rusage[mem=64GB] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=no" \
    -g /saumik/limit100 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python _3_eval_models.py" $NODE $STUDENT $TEACHER
    sleep 0.1
done
done
done
# -o /storage1/fs1/chien-ju.ho/Active/chess/logs/${TEACH}_${CK}.%J \

    # echo $STUDENT $CURR $CK $TEACH
    # bsub -n 8 \
    # -q general \
    # -m general \
    # -G compute-chien-ju.ho \
    # -J ${TEACH}_${CK} \
    # -M 128GB \
    # -N \
    # -u saumik@wustl.edu \
    # -o /storage1/fs1/chien-ju.ho/Active/chess/logs/${TEACH}_${CK}.%J \
    # -R "rusage[mem=128GB] span[hosts=1] select[gpuhost&&hname!='compute1-exec-201.ris.wustl.edu']" \
    # -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=no" \
    # -g /saumik/limit100 \
    # -a "docker(saumikn/chesstrainer)" \
    # "cd ~/chesstrainer && /opt/conda/bin/python experiments_sn.py" ${STUDENT} ${CURR} ${TEACH} ${CK}
    # sleep 0.1
