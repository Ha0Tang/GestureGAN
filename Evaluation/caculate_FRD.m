%close all; clear all; clc
function final_fvd_score=caculate_FRD(Image_folder_name)
Image_folder=strcat(Image_folder_name,'/images');
%Image_folder='image';
Image =  dir( Image_folder );  
fake_image=[];
real_image=[];
fvd_score=[];
j=1;
k=1;
for i = 3 : length( Image )
    image_name=Image( i ).name;
    if contains(image_name, 'fake_B')
        fake_image{j}=image_name;
        j=j+1;
    elseif contains(image_name, 'real_B')
        real_image{k}=image_name;
        k=k+1;
    end  
end

net = resnet50;
sz = net.Layers(1).InputSize;
layer = 'fc1000';
ground_truth_feature_final = [];
generated_feature_final = [];
for o=1:length(fake_image)
    fprintf('%d / %d \n', o, length(fake_image));
	generated_image_path = fullfile( Image_folder, fake_image{o});
    generated_image=imread(generated_image_path);
%     imshow(generated_image)
	ground_image_path=fullfile( Image_folder, real_image{o});
	ground_truth_image=imread(ground_image_path);
%     imshow(real_image)
    
    ground_truth_image = imresize(ground_truth_image, [sz(1) sz(2)]);
    generated_image  = imresize(generated_image, [sz(1) sz(2)]);
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
display(Image_folder)
final_fvd_score=mean(fvd_score)


