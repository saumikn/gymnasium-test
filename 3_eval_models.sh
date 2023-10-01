for MAP_SIZE in 16
# for MAP_SIZE in 12
do
for MYOPIC in {1..40}
# for MYOPIC in {1..30}
do
    echo $MAP_SIZE $MYOPIC
    bsub -n 1 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${MAP_SIZE}_${MYOPIC} \
    -M 64GB \
    -N \
    -u saumik@wustl.edu \
    -o /home/n.saumik/gymnasium-test/tmp/${MAP_SIZE}_${MYOPIC}.%J \
    -R "rusage[mem=64GB] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:j_exclusive=yes" \
    -g /saumik/limit10 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python 3_eval_models.py" ${MAP_SIZE} ${MYOPIC}
    sleep 0.1
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
