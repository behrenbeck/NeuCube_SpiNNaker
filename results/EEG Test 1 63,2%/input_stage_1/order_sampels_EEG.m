%% Load class samples

clear
clc

sam_vec_1 = [];
sam_vec_2 = [];

labels = csvread('tar_class_labels_EEG.txt');

for i=1:399
    sam = csvread(strcat('sam_',int2str(i),'.csv'));
    if labels(i) == 1
        sam_vec_1 = [sam_vec_1 ; sam];
    elseif labels(i) == 2
        sam_vec_2 = [sam_vec_2 ; sam];
    else
        disp('Wrong label at ' + i)
    end
end


%% Create training samples

final = [];
labels = [] ;
share_1 = 19;
share_2 = 19;
idx = 1;
for i=1:5

    final = [final ; sam_vec_1(idx:idx+3*share_1-1,1:250)];
    labels = [labels ; 1*ones(share_1,1)];
    final = [final ; sam_vec_2(idx:idx+3*share_1-1,1:250)];
    labels = [labels ; 2*ones(share_1,1)];
    idx = idx + 3*share_1;
    
    final = [final ; sam_vec_1(idx:idx+3*share_2-1,1:250)];
    labels = [labels ; 1*ones(share_2,1)];
    final = [final ; sam_vec_2(idx:idx+3*share_2-1,1:250)];
    labels = [labels ; 2*ones(share_2,1)];
    idx = idx + 3*share_2;

end

%final = flipud(final);
%final = [final(289:end,:); final(1:288,:)];
%% Save Samples

for i=1:380
    sam = final(1+(i-1)*3:i*3,:);
    csvwrite(strcat('sam_',int2str(i),'.csv'),sam)
end

csvwrite('tar_class_labels_EEG.txt',labels)