clear all; clc; close all;

%Import data
file = readtable('Breast_cancer_data.csv');

file.Properties.VariableNames = {'id','diagnosis','radius_mean','texture_mean',...
    'perimeter_mean','area_mean','smoothness_mean','compactness_mean',...
    'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'...,
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',...
    'compactness_se','concavity_se','concave points_se','symmetry_se',...
    'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',...
    'area_worst','smoothness_worst','compactness_worst','concavity_worst',...
    'concave points_worst','symmetry_worst','fractal_dimension_worst'};

%Remove ID column as it is not contributin to the predictive model 
file(:,1) = [];

%Split the file into X(predicotrs) and Y(labels) respectively 
X_table = file(:,[2:31]);
Y_table = file(:,1);

X_table.Properties.VariableNames={'RM','TM',...
    'PM','AM','SoM','CM','CcM','CpM','SyM','FDM','RS','TS','PS','AS','SoS',...
    'CS','CcS','CpS','SyS','FDS','RW','TW','PW','AW','SoW','CW','CcW',...
    'CpW','SyW','FDW'};

%Change the format of X and Y 
X  = cell2mat(table2cell(X_table));
Y  = table2cell(Y_table);

Class_names = {'B','M'};

num_rows = length(X(:, 1));

%Split the data into training and testing for X and Y respectively 
m   = num_rows;
p   = 0.7;
n   = round(p*m);
idx = randperm(m);

Training   = file(idx(1:n), :);
Testing    = file(idx(n+1:end), :);

X_Training = X(idx(1:n), :);
X_Testing  = X(idx(n+1:end), :);

Y_Training = Y(idx(1:n), :);
Y_Testing  = Y(idx(n+1:end), :);

%Compute the prior
prob_first_class = sum(strcmp(Y, Class_names(1)))/num_rows;
sample_prior = [prob_first_class, 1-prob_first_class];




