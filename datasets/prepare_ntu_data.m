clear all;close all;clc;

image_folder='./datasets/ntu_image_skeleton/ntu_image';
skeleton_folder='/datasets/ntu_image_skeleton/ntu_skeleton';

txt_file ='./datasets/ntu_split/ntu_test.txt';
save_folder='./datasets/ntu/test';

// Uncomment for training data
// txt_file ='./datasets/ntu_split/ntu_train.txt';
// save_folder='./datasets/ntu/train';

if ~isfolder(save_folder)
    mkdir(save_folder)
end

image_name = importdata(txt_file);

for i=1:length(image_name)
    fprintf('%d / %d \n', i, length(image_name));
    index=strfind(image_name{i},'color');
    if isempty(strfind(image_name{i}, '_lr'))
        A_path=fullfile(image_folder,strcat(image_name{i}(1:index(2)-2),'.png'));
        B_path=fullfile(image_folder,image_name{i}(index(2):length(image_name{i})));
        C_path=fullfile(skeleton_folder,strcat(image_name{i}(1:index(2)-2),'.png'));
        D_path=fullfile(skeleton_folder,image_name{i}(index(2):length(image_name{i})));   
        A=imread(A_path);
        B=imread(B_path);
        C=imread(C_path);
        D=imread(D_path);
        
        image_all=[A,B,C,D];
    else 
        A_path=fullfile(image_folder,strcat(image_name{i}(1:index(2)-2),'.png'));
        B_path=fullfile(image_folder,strcat(image_name{i}(index(2):length(image_name{i})-10),'.png'));
        C_path=fullfile(skeleton_folder,strcat(image_name{i}(1:index(2)-2),'.png'));
        D_path=fullfile(skeleton_folder,strcat(image_name{i}(index(2):length(image_name{i})-10),'.png'));   
        A=imread(A_path);
        B=imread(B_path);
        C=imread(C_path);
        D=imread(D_path);
        image_all=[fliplr(A),fliplr(B),fliplr(C),fliplr(D)];
    end
    imwrite(image_all,fullfile(save_folder,image_name{i}));
 
end

