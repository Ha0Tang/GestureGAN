python test.py --dataroot ./datasets/cvusa \
        --name cvusa_gesturegan_onecycle \
        --model gesturegan_onecycle \
        --which_model_netG resnet_9blocks \
        --which_direction AtoB \
        --dataset_mode aligned \
        --norm instance \
        --gpu_ids 0,1 \
        --batchSize 12 \
        --loadSize 286 \
        --fineSize 256 \
        --no_flip \
        --how_many 1000000000000000000 \
        --saveDisk \
        --which_epoch 25




