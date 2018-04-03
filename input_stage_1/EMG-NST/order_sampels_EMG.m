%% Load class samples

clear
clc

sam_vec_1 = [];
for i=1:36
    sam = csvread(strcat('sam_',int2str(i),'.csv'));
    sam_vec_1 = [sam_vec_1 ; sam];
end

sam_vec_2 = [];
for i=37:72
    sam = csvread(strcat('sam_',int2str(i),'.csv'));
    sam_vec_2 = [sam_vec_2 ; sam];
end

sam_vec_3 = [];
for i=73:108
    sam = csvread(strcat('sam_',int2str(i),'.csv'));
    sam_vec_3 = [sam_vec_3 ; sam];
end

sam_vec_4 = [];
for i=109:144
    sam = csvread(strcat('sam_',int2str(i),'.csv'));
    sam_vec_4 = [sam_vec_4 ; sam];
end

%% Create training samples

final = [];
share_1 = 6;
share_2 = 6;
idx = 1;
for i=1:3

    final = [final ; sam_vec_1(idx:idx+4*share_1-1,:)];
    final = [final ; sam_vec_2(idx:idx+4*share_1-1,:)];
    final = [final ; sam_vec_3(idx:idx+4*share_1-1,:)];
    final = [final ; sam_vec_4(idx:idx+4*share_1-1,:)];
    idx = idx + 4*share_1;
    
    final = [final ; sam_vec_1(idx:idx+4*share_2-1,:)];
    final = [final ; sam_vec_2(idx:idx+4*share_2-1,:)];
    final = [final ; sam_vec_3(idx:idx+4*share_2-1,:)];
    final = [final ; sam_vec_4(idx:idx+4*share_2-1,:)];
    idx = idx + 4*share_2;

end
%final = flipud(final);
%final = [final(289:end,:); final(1:288,:)];
%% Save Samples

for i=1:144
    sam = final(1+(i-1)*4:i*4,:);
    csvwrite(strcat('sam_',int2str(i),'.csv'),sam)
end