rng(1)

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
X  = cell2mat(table2cell(file(:,[2:31])));
Y  = table2cell(file(:,1));

Class_names = {'B','M'};

num_rows = length(X(:, 1));

%Split the data into training and testing for X and Y respectively 
m   = num_rows;
p   = 0.5;
n   = round(p*m);
idx = randperm(m);

X_Training = X(idx(1:n), :);
X_Testing  = X(idx(n+1:end), :);

Y_Training = Y(idx(1:n), :);
Y_Testing  = Y(idx(n+1:end), :);

%Compute the prior
prob_first_class = sum(strcmp(Y, Class_names(1)))/num_rows;
sample_prior = [prob_first_class, 1-prob_first_class];

% 10-fold cross validation
rng(1)

widths         = 0.1:0.1:0.9;
prior_offsets  = [-0.05 -0.03 0 0.03 0.05];
l              = round(p*n);
distributions   = ["kernel","mvmn","normal"];

names      = cell(length(prior_offsets),length(widths) + 2);
accuracies = zeros(length(prior_offsets), length(widths) + 2, 10);
losses     = zeros(length(prior_offsets), length(widths) + 2, 10);

tic
%Prior 
for i = 1:length(prior_offsets)
    prior = [sample_prior(1) + prior_offsets(i); sample_prior(2) - prior_offsets(i)];
    
    for iteration=1:10
        idx = randperm(n);
        X_Training_set   = X_Training(idx(1:l),:);
        Y_Training_set   = Y_Training(idx(1:l),:);
        X_Validation_set = X_Training(idx(l+1:end),:);
        Y_Validation_set = Y_Training(idx(l+1:end),:);  
        
        %Distributions
        for j = 1:length(distributions)
            
            % Kernel
            if distributions(j) == "kernel"
                
                for k = 1:length(widths)
                    model = fitcnb(X_Training_set,Y_Training_set,...
                        'ClassNames', Class_names,...
                        'DistributionNames', distributions(j),...
                        'Prior', prior,...
                        'Width', widths(k));
                    [bayes_predicted, score_train] = model.predict(X_Validation_set);
                    % Create a confusion matrix
                    confusion_matrix = confusionmat(Y_Validation_set, bayes_predicted);
                    % Obtain the error rate of the model
                    bayes_accuracy = trace(confusion_matrix)/sum(confusion_matrix(:));
                    % Obtain the loss of the model
                    bayes_loss = loss(model, X_Validation_set, Y_Validation_set);
                    % Storing the error rate, distribution name and the prior for each model
                    accuracies(i,k, iteration) = bayes_accuracy;
                    losses(i,k, iteration) = bayes_loss;
                end 
            
            else
                idx = randperm(n);
                X_Training_set   = X_Training(idx(1:l),:);
                Y_Training_set   = Y_Training(idx(1:l),:);
                X_Validation_set = X_Training(idx(l+1:end),:);
                Y_Validation_set = Y_Training(idx(l+1:end),:);  
                
                model = fitcnb(X_Training_set,Y_Training_set,...
                        'ClassNames', Class_names,...
                        'DistributionNames', distributions(j),...
                        'Prior', prior);
                [bayes_predicted, score_train] = model.predict(X_Validation_set);
                % Create a confusion matrix
                confusion_matrix = confusionmat(Y_Validation_set, bayes_predicted);
                % Obtain the error rate of the model
                bayes_accuracy = trace(confusion_matrix)/sum(confusion_matrix(:));
                % Obtain the loss of the model
                bayes_loss = loss(model, X_Validation_set, Y_Validation_set);
                % Storing the error rate, distribution name and the prior for each model
                accuracies(i,length(widths)+j-1, iteration) = bayes_accuracy;
                losses(i,length(widths)+j-1, iteration) = bayes_loss;  
            end 
        end 
    end 
    
    for j = 1:length(widths)
        names{i, j} = strcat(num2str('kernel', num2str(widths(j))));
    end
    
    names{i, length(widths)+1} ='mvmn';
    names{i, length(widths)+2} ='normal';

end 
toc

%Store the from 10-fold cross validation results in a table 
results    = mean(accuracies,3);
dist_names = cell(1, length(widths) + 2);

for j = 1:length(widths)
    dist_names{j} = strcat('kernel', num2str(widths(j)));
end

dist_names{length(widths)+1} = 'mvmn';
dist_names{length(widths)+2} = 'normal';

Results = array2table(results);
Results.Properties.VariableNames = dist_names;

prior_offsets = array2table(prior_offsets');
Results = [prior_offsets,Results];

Results.Properties.VariableNames={'Prior','kernel0.1 ','kernel0.2','kernel0.3',...
    'kernel0.4','kernel0.5','kernel0.6','kernel0.7','kernel0.8','kernel0.9',...
    'mvmn','normal'};

%Find the highest model accuracy 
[~,maxidx] = max(Results.normal);
Results(maxidx,:)

%Train a new model on the test set with the best parameters 
rng(1)

Bayes_model = fitcnb(X_Training,Y_Training,'ClassNames',Class_names,...
    'DistributionNames','normal','Prior',sample_prior);

Info = table(Y_Testing,X_Testing,'VariableNames',{'TrueLabel','PredictedLabel'});

[Best_Bayes_predicted,Scores] = Bayes_model.predict(X_Testing);
[Confusion_matrix,~] = confusionmat(Y_Testing,Best_Bayes_predicted);

%Plot the confusion matrix
figure('Name','Confusion Matrix for Naive Bayes Model')
confusionchart(Confusion_matrix)

%Calculate the accuracy and the loss of the model 
Accuracy_Bayes = trace(Confusion_matrix)/sum(Confusion_matrix(:));
Loss = loss(Bayes_model,X_Testing,Y_Testing);

%Compare the result with Baysian optimised model 
Opt_Bayes_model = fitcnb(X_Training,Y_Training,'ClassNames',Class_names,'Prior',...
    sample_prior,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));





