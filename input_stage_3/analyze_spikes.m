%% add up

clc

labels = textread('tar_class_labels.txt');

spikes_class_1 = [];
spikes_class_2 = [];
for i = 1:size(labels)
    spike_train = textread(strcat('reservoir_spikes_sam_',int2str(i),'.txt'));
    if labels(i) == 1
        spikes_class_1 = [spikes_class_1 ; spike_train];
    elseif labels(i) == 2
        spikes_class_2 = [spikes_class_2 ; spike_train];
    end
end

figure
subplot(2,1,1)
histogram(spikes_class_1(:,1),linspace(1,1471,1471))
title('class 1')
axis([0 1471 0 3000])
subplot(2,1,2)
histogram(spikes_class_2(:,1),linspace(1,1471,1471))
title('class 2')
axis([0 1471 0 3000])

%% isolated sample

clc

figure
spike_train = textread('reservoir_spikes_sam_12.txt');
subplot(2,1,1)
histogram(spike_train(:,1),linspace(1,1471,1471))
axis([0 1471 0 200])

%get first spikes
first_spike = zeros(1471,1);
n_idx = spike_train(1,1);
first_spike(n_idx) = spike_train(1,2);
for i = 1: size(spike_train,1)
    spike = spike_train(i,:);
    if spike(1) ~= n_idx
        n_idx = spike(1);
        first_spike(n_idx) = spike(2);
    end
end
% calculate importance of synapses
importance = zeros(1471,1);
for i = 1:1471
    if first_spike(i) ~= 0
        importance(i) = 1000 - first_spike(i);
    end
end

subplot(2,1,2)
stem(importance,'b.')
axis([0 1471 0 1000])
