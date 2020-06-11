close all; clear all; clc

Image_folder_name ='./data';

real_image_folder=strcat(Image_folder_name,'/realimage');
fake_image_folder=strcat(Image_folder_name,'/fakeimage');

Image =  dir( real_image_folder );  

fvd_score=[];
net = resnet50;

sz = net.Layers(1).InputSize;
layer = 'fc1000';
ground_truth_feature_final = [];
generated_feature_final = [];

for o=1:length(Image)
    fprintf('%d / %d \n', o, length(Image)-2);
    if( isequal( Image(o).name, '.' ) || isequal( Image(o).name, '..' ))    
        continue;
    end
    
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


