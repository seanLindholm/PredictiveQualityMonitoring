%% Get data
dt_mixed = readmatrix("Mixed_transform_noClass_unNorm");
dt_mixed = dt_mixed(:,4:end);

f_mixed = ones(size(dt_mixed,1),1);

%% use dea
io_mixed = dea(f_mixed,dt_mixed,'orient','io').eff;


%% Some simple analytic

min(io_mixed)
max(io_mixed)
fileID = fopen('eff_mixed.csv','w');
fprintf(fileID,'dea_eff\r\n');
fprintf(fileID,'%6.2f\r\n',io_mixed);
fclose(fileID);




