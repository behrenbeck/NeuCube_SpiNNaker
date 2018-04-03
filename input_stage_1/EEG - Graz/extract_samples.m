%% load file
clc
clear

subject = 'B09T';
start_offset = 2;          %start of time window after imagination onset (t=4)in seconds
duration = 1;              %duration in seconds
filter_freq = [0.1 60];    %lower and upper frequency range for bandpass filter

load(strcat('C:\Users\Jan\Code\EEG-brain-signals-decoding\NeuCube_spiNNaker\NeuCube_Code\input_stage_1\EEG - Graz\Data GDF\',subject,'.mat'));

%% extract data

data_class_1 = [];
data_class_2 = [];

for session = 1:size(data,2)
    data_session = data{1,session};
    X_session = data_session.X;
    y_session = data_session.y;
    trial_session = data_session.trial;
    artifacts_session = data_session.artifacts;
    frequency_session = data_session.fs;
    X_session_filt = bandpass(X_session(:,1:3),filter_freq,frequency_session);
%     figure
%     subplot(3,1,1)
%     plot(X_session(200:500,1:3))
%     subplot(3,1,2)
%     plot(X_session_filt(200:500,:))
%     subplot(3,1,3)
%     plot(X_session(200:500,1:3)-X_session_filt(200:500,:))
    
    for trial=1:size(trial_session)
        if ~ artifacts_session(trial)
            start_time = trial_session(trial)+start_offset*frequency_session; 
            sample = X_session_filt(start_time:start_time+duration*frequency_session-1,1:3);
            %sample = X_session(start_time:start_time+duration*frequency_session-1,1:3);
            sample_norm = normalize(sample);
            if y_session(trial) == 1
                data_class_1 = [data_class_1 ; sample_norm];
            elseif y_session(trial) == 2
                data_class_2 = [data_class_2 ; sample_norm];
            else
                disp('Faaaaaaaaaail: Unknown class')
            end
        end      
    end
    
end

%% save samples
class_labels = [];
share = 5;

for count = 1 : min(floor(size(data_class_1,1)/frequency_session/share),floor(size(data_class_2,1)/frequency_session/share))
    %class 1
    for i = 1:share
        index = i + (count-1)*2*share;
        sample = data_class_1(index:index+frequency_session-1,:);
        filename = strcat('sam_',num2str(index),'.csv');
        csvwrite(filename,sample');
    end
    class_labels = [class_labels ; 1*ones(share,1)];
    %class 2
    for i = 1:share
        index = i + (count-1)*2*share;
        sample = data_class_1(index:index+frequency_session-1,:);
        filename = strcat('sam_',num2str(index+5),'.csv');
        csvwrite(filename,sample');
    end
    class_labels = [class_labels ; 2*ones(share,1)];
end
fileID = fopen(strcat('tar_class_labels_',subject,'.txt'),'w');
fprintf(fileID,strcat('%d',' \n'),class_labels);
fclose(fileID);
