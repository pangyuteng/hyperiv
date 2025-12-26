
docker run -it -p 8888:8888 \
--shm-size=256m --gpus all \
-w $PWD -v /mnt:/mnt pangyuteng/hyperiv bash


jupyter notebook --ip=* --allow-root