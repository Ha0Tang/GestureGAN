% close all; clear all; clc

Image_folder_name ='/data4/hao/retrieved_data/results_from_other/all_senz3d_cyc01_id001_vgg1000_l1800_twocyc_286_gesture/test_latest';
% Image_folder_name ='/Users/hao/Desktop/iccv19_show/results/all_ntu_cyc01_id0001/test_latest';

real_image_folder=strcat(Image_folder_name,'/realimage_B');
fake_image_folder=strcat(Image_folder_name,'/fakeimage_B');

Image =  dir( real_image_folder );  

fvd_score=[];
net = resnet50;

sz = net.Layers(1).InputSize;
layer = 'fc1000';
ground_truth_feature_final = [];
generated_feature_final = [];
for o=3:length(Image)
    fprintf('%d / %d \n', o-2, length(Image)-2);
	generated_image_path = fullfile( fake_image_folder, Image( o ).name);
	generated_image=imread(generated_image_path);
	ground_image_path=fullfile( real_image_folder, Image( o ).name);
	ground_truth_image=imread(ground_image_path);

	ground_truth_feature = activations(net, ground_truth_image, layer);
	generated_feature = activations(net, generated_image, layer);
    
    for i=1:length(ground_truth_feature)
        ground_truth_feature_final(1,i) = ground_truth_feature(:,:,i);
    end
    for i=1:length(generated_feature)
        generated_feature_final(1,i) = generated_feature(:,:,i);
    end
    
    [fvd, cSq] = DiscreteFrechetDist(ground_truth_feature_final',generated_feature_final');
      
    fvd_score(o)=fvd;
end
display(Image_folder_name)
final_fvd_score=mean(fvd_score)


