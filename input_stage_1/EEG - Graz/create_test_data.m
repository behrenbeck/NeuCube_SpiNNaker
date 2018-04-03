%% Jan Behrenbeck - 04.12.2017

% Create test samples and label file

clear all
close all

%% Create samples

n = 100;

gauss = gausswin(n);
sinus = (sin(linspace(0,2*pi,n))'+sin(10*linspace(0,2*pi,n))'+2*sin(4*linspace(0,2*pi,n))')/8+0.5;
cosin = (cos(linspace(0,2*pi,n))')/2+0.5;
linea = linspace(0,1,n)';

%% Visualize samples

figure
subplot(5,1,1);plot(gauss);title('Gaussian window')
subplot(5,1,2);plot(sinus);title('Sine')
subplot(5,1,3);plot(cosin);title('Cosine')
subplot(5,1,4);plot(linea);title('Linear')
subplot(5,1,5);plot(csvread('norm_EEG.csv'));title('Linear')
suptitle('Test samples')

%% Create target class labels

labels = [1 0 0 0 0]';

%% Save Files

csvwrite('tar_class_labels.csv',labels)
csvwrite('sam1_gauss.csv',gauss)
csvwrite('sam2_sinus.csv',sinus)
csvwrite('sam3_cosin.csv',cosin)
csvwrite('sam4_linea.csv',linea)

