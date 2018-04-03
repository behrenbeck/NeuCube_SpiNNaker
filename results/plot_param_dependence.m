%% conn_prob
acc = [0.47   , 0.5    , 0.445  , 0.485  , 0.43   , 0.5    ]; %0.09 - 0.14
var = [0.0073 , 0.005  , 0.0036 , 0.0045 , 0.0073 , 0.0061 ];
par_start = 0.09;
par_end   = 0.14;

%% weight
acc = [0.435  , 0.51   , 0.435  , 0.445  , 0.475  , 0.515  ]; %3.8 - 4.3
var = [0.0122 , 0.0127 , 0.0067 , 0.0047 , 0.0135 , 0.0034 ];
par_start = 3.8;
par_end   = 4.3;

%% inh_split
acc = [0.53   , 0.425  , 0      , 0.475  , 0.49   , 0.48   ]; %0.15 - 0.4
var = [0.0073 , 0.0079 , 0      , 0.0063 , 0.0082 , 0.0062 ];
par_start = 0.15;
par_end   = 0.4;

%% plot

par = linspace(par_start,par_end,6);

figure
errorbar(par,acc,sqrt(var),'r*')
xlabel('parameter name')
ylabel('mean classification accuracy')
axis([0.99*par_start 1.01*par_end 0 1])
