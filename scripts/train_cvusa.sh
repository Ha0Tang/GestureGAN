export CUDA_VISIBLE_DEVICES=0,1;
python train.py --dataroot ./datasets/cvusa \
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
	--cyc_L1 0.1 \
	--lambda_identity 100 \
	--lambda_feat 100 \
	--display_id 0 \
	--niter 15 \
	--niter_decay 15 \
        # --continue_train --which_epoch 6 --epoch_count 7
