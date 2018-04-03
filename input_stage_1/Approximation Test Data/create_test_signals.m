%% Create and plot signals

x = linspace (0,1,251);
cosinus = 0.4*cos(2*pi*x)+0.5;
sinus = 0.4*sin(13*pi*x)+0.5;
idle = [zeros(1,240) 0.5*ones(1,11)];
linear = 0.7*x;

figure
subplot(4,1,1)
plot(1000*x,cosinus)
ylabel('norm EEG')
axis([0 1000 0 1])
title('Low-Hz Cosine')
subplot(4,1,2)
plot(1000*x,sinus)
ylabel('norm EEG')
axis([0 1000 0 1])
title('High-Hz Sine')
subplot(4,1,3)
plot(1000*x,linear)
ylabel('norm EEG')
axis([0 1000 0 1])
title('Linear increasing')
subplot(4,1,4)
plot(1000*x,idle)
xlabel('time [ms]')
ylabel('norm EEG')
axis([0 1000 0 1])
title('Idle state')
suptitle('Test Functions for Approximation Testing')


%% Save to files

distribute = true;
if distribute == true
    sam1 = [cosinus;idle;idle];
    sam2 = [idle;sinus;idle];
    sam3 = [idle;idle;linear];
    sam4 = [idle;idle;idle];
else
    sam1 = [cosinus;linear;sinus];
    sam2 = [sinus;cosinus;linear];
    sam3 = [linear;sinus;cosinus];
    sam4 = [idle;idle;idle];
end

csvwrite('sam_1.csv',sam1)
csvwrite('sam_2.csv',sam2)
csvwrite('sam_3.csv',sam3)
csvwrite('sam_4.csv',sam4)