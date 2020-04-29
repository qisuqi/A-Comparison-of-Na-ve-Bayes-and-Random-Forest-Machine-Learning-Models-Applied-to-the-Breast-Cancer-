clear all; clc; close all;

run('Data_preparation.m')

% Create Naive Bayes Model
%Split the data to 3 subsets 
attr_mean  = X(:,[1:10]);
attr_se    = X(:,[11:20]);
attr_worst = X(:,[21:30]);

% Test if each subset would alter the model prediction 
attr_mean_training  = attr_mean(idx(1:n), :);
attr_mean_testing   = attr_mean(idx(n+1:end), :);

attr_se_training    = attr_se(idx(1:n), :);
attr_se_testing     = attr_se(idx(n+1:end), :);

attr_worst_training = attr_worst(idx(1:n), :);
attr_worst_testing  = attr_worst(idx(n+1:end), :);

attr_mean_bayes_model      = fitcnb(attr_mean_training, Y_Training, 'ClassNames', Class_names, 'Prior', sample_prior);
attr_mean_bayes_predicted  = attr_mean_bayes_model.predict(attr_mean_testing);
attr_mean_confusion_matrix = confusionmat(Y_Testing, attr_mean_bayes_predicted); 
attr_mean_bayes_accuracy   = trace(attr_mean_confusion_matrix)/sum(attr_mean_confusion_matrix(:));
 
attr_se_bayes_model      = fitcnb(attr_se_training, Y_Training, 'ClassNames', Class_names, 'Prior', sample_prior);
attr_se_bayes_predicted  = attr_se_bayes_model.predict(attr_se_testing);
attr_se_confusion_matrix = confusionmat(Y_Testing, attr_se_bayes_predicted); 
attr_se_bayes_accuracy   = trace(attr_se_confusion_matrix)/sum(attr_se_confusion_matrix(:));

attr_worst_bayes_model      = fitcnb(attr_worst_training, Y_Training, 'ClassNames', Class_names, 'Prior', sample_prior);
attr_worst_bayes_predicted  = attr_worst_bayes_model.predict(attr_worst_testing);
attr_worst_confusion_matrix = confusionmat(Y_Testing, attr_worst_bayes_predicted); 
attr_worst_bayes_accuracy   = trace(attr_worst_confusion_matrix)/sum(attr_worst_confusion_matrix(:));
 
sample_bayes_model      = fitcnb(X_Training, Y_Training, 'ClassNames', Class_names, 'Prior', sample_prior);
sample_bayes_predicted  = sample_bayes_model.predict(X_Testing);
sample_confusion_matrix = confusionmat(Y_Testing, sample_bayes_predicted);
sample_bayes_accuracy   = trace(sample_confusion_matrix)/sum(sample_confusion_matrix(:));

cvm_bayes_model_1 = fitcnb(X_Training, Y_Training, 'ClassNames', Class_names, 'Prior', sample_prior, 'CrossVal', 'on');
cvm_bayes_model_2 = fitcecoc(X_Training, Y_Training, 'ClassNames', Class_names, 'CrossVal', 'on', 'Learners', templateNaiveBayes());
 
cvm_bayes_error_1 = kfoldLoss(cvm_bayes_model_1, 'LossFun', 'ClassifErr');
cvm_bayes_error_2 = kfoldLoss(cvm_bayes_model_2, 'LossFun', 'ClassifErr');

errors = [
     "attr_mean_bayes_error"  attr_mean_bayes_accuracy;
     "attr_se_bayes_error"    attr_se_bayes_accuracy;
     "attr_worst_bayes_error" attr_worst_bayes_accuracy;
     "sample_bayes_error"     sample_bayes_accuracy;
     "cvm1"                   1 - cvm_bayes_error_1;
     "cvm2"                   1 - cvm_bayes_error_2;
];


% 10-fold cross validation
widths         = 0.1:0.1:0.9;
prior_offsets  = [-0.05 -0.03 0 0.03 0.05];
l              = round(p*n);
distributions  = ["kernel","mvmn","normal"];

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

%Plot the final result for comparison 
figure('Name','Visualisation on Final Results of NB')
x = categorical({'kernel0.1 ','kernel0.2','kernel0.3',...
    'kernel0.4','kernel0.5','kernel0.6','kernel0.7','kernel0.8','kernel0.9',...
    'mvmn','normal'})
y = results
bar(x,y)
ylim([0.55 1])
title('10-fold Cross Validation Results of NB','FontSize',24)
legend('Prior-0.05','Prior-0.03','Prior','Prior+0.03','Prior+0.05','Location','northwest')

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




