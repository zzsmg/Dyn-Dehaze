clc;
clear;
close all;
for i=41:120
    haze_path = sprintf('/data/Pytorch_Porjects/DWT-FFC/datasets/combined_dataset/Train/haze/%03d_haze.png',i);
    prior_path = sprintf('/data/Pytorch_Porjects/DWT-FFC/datasets/combined_dataset/Train/prior/%03d_prior.png',i);
    haze=imread(haze_path);
    haze=double(haze)./255;
    dehaze=run_cnn(haze);
    imwrite(dehaze,prior_path)
end
