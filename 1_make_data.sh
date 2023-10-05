for MAP_SIZE in 12
do
for GROUP in 771 772 776 777 778 781 869
# for GROUP in {0..999}
do
    echo $MAP_SIZE $GROUP
    bsub -n 16 \
    -q general \
    -m general \
    -G compute-chien-ju.ho \
    -J ${MAP_SIZE}_${GROUP} \
    -M 64GB \
    -N \
    -u saumik@wustl.edu \
    -R "rusage[mem=64GB] span[hosts=1]" \
    -g /saumik/limit100 \
    -a "docker(saumikn/chesstrainer:gym)" \
    "cd ~/gymnasium-test && /opt/conda/bin/python 1_make_data.py" ${MAP_SIZE} ${GROUP}
    sleep 0.1
done
done
# -o /storage1/fs1/chien-ju.ho/Active/chess/logs/${TEACH}_${CK}.%J \
