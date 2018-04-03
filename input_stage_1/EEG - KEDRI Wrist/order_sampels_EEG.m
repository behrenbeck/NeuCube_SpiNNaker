%% Load class samples

clear
clc

sam_vec_1 = [];
sam_vec_2 = [];
sam_vec_3 = [];

labels = csvread('tar_class_labels_EEG.csv');

for i=1:60
    sam = csvread(strcat('sam',int2str(i),'_eeg.csv'));
    sam = (sam-min(sam))./(max(sam)-min(sam));    
    
    if labels(i) == 1
        sam_vec_1 = [sam_vec_1 ; sam'];
    elseif labels(i) == 2
        sam_vec_2 = [sam_vec_2 ; sam'];
    elseif labels(i) == 3
        sam_vec_3 = [sam_vec_3 ; sam'];
    else
        disp('Wrong label at ' + i)
    end
end


%% Create training samples

final = [];
labels = [] ;
share_1 = 5;
share_2 = 5;
idx = 1;
for i=1:2

    final = [final ; sam_vec_1(idx:idx+14*share_1-1,:)];
    labels = [labels ; 1*ones(share_1,1)];
    final = [final ; sam_vec_2(idx:idx+14*share_1-1,:)];
    labels = [labels ; 2*ones(share_1,1)];
    final = [final ; sam_vec_3(idx:idx+14*share_1-1,:)];
    labels = [labels ; 3*ones(share_1,1)];
    idx = idx + 14*share_1;
    
    final = [final ; sam_vec_1(idx:idx+14*share_2-1,:)];
    labels = [labels ; 1*ones(share_2,1)];
    final = [final ; sam_vec_2(idx:idx+14*share_2-1,:)];
    labels = [labels ; 2*ones(share_2,1)];
    final = [final ; sam_vec_3(idx:idx+14*share_2-1,:)];
    labels = [labels ; 3*ones(share_2,1)];
    idx = idx + 14*share_2;

end

%final = flipud(final);
%final = [final(289:end,:); final(1:288,:)];
%% Save Samples

for i=1:60
    sam = final(1+(i-1)*14:i*14,:);
    csvwrite(strcat('sam_',int2str(i),'.csv'),sam)
end

csvwrite('tar_class_labels.txt',labels)