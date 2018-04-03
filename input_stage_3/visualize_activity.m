 spikes = textread('reservoir_spikes_sam_1.txt');
 
 neurons = zeros(1471,1);
 
 for i=1:size(spikes,1)
     neurons(spikes(i,1)) = neurons(spikes(i,1)) + 1;
 end
 
 neurons = sqrt(neurons / max(neurons));
 histogram(neurons)
 
 figure 
 hold on 
 for n = 1:1471
     if neurons(n) == 0
         color = [0.9 0.9 0.9];
     else
         color = [neurons(n) 0 0];
     end
    scatter3(pos(n,1),pos(n,2),pos(n,3),'MarkerEdgeColor',color,'LineWidth',neurons(n)+0.1)
 end
 