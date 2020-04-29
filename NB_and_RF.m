clear all; clc; close all;

rng(1)

run('Data_preparation.m')

%Create Naive Bayes model with the best parameters 
Bayes_model_final=fitcnb(X_Training,Y_Training,'ClassNames',Class_names,...
'DistributionNames','normal','Prior',sample_prior);

[Best_Bayes_predicted,Scores]=Bayes_model_final.predict(X_Testing);
[Bayes_Confusion_matrix,~]=confusionmat(Y_Testing,Best_Bayes_predicted);

%Plot the confusion matrix
figure('Name','Confusion Matrix for Naive Bayes Model')
confusionchart(Bayes_Confusion_matrix,'Title','NB Confusion Matrix','FontSize',24)

%Calculate the accuracy and the loss of the model 
Bayes_accuracy_final=100*(trace(Bayes_Confusion_matrix)/sum(Bayes_Confusion_matrix(:)));
Bayes_loss_final=loss(Bayes_model_final,X_Testing,Y_Testing);

%Compute the False Positive and True Positive of the Naive Bayes model 
[Bayes_FP,Bayes_TP,T,Bayes_AUC]=perfcurve(Y_Testing,Scores(:,2),'M');

clc

%Create Random Forest model with the best parameters 
rng(1)

RF_model_final = TreeBagger(80, X_Training,Y_Training, 'OOBPrediction',... 
'on', 'MinLeafSize', 6, 'NumPredictorsToSample', 8); 

%Test the model using predict and the testing dataset 
[RF_predicted_labels,scores]= RF_model_final.predict(X_Testing);
[RF_Confusion_matrix,order]= confusionmat(Y_Testing,RF_predicted_labels);

%Plot the confusion matrix 
figure
confusionchart(RF_Confusion_matrix,'Title','RF Confusion Matrix','FontSize',24)
       
%Calculate the accuracy and the loss of the model 
RF_accuracy_final = 100*sum(diag(RF_Confusion_matrix))./sum(RF_Confusion_matrix(:)) ; 
RF_error_final =  oobError(RF_model_final, 'Mode', 'Ensemble') ; 

%Find the column which the malignant cases are in 
MPosition = find(strcmp('M',RF_model_final.ClassNames)) ; 

%Compute the False Positive and True Positive of the Random Forest model 
[RF_FP,RF_TP,T,RF_AUC] = perfcurve(Y_Testing,scores(:,MPosition),'M');

%Plot the ROC curve for both Naive Bayes and Random Forest
figure
plot(RF_FP,RF_TP)
hold on 
plot(Bayes_FP,Bayes_TP)
legend('Random Foreset','Naive Bayes','Location','Best','Location','northeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve for Naive Bayes and Random Forest','FontSize',24)
hold off

%Assign the True Positive, True Negative, False Postive and False Negative of
%the confusion matrix of both models 
Bayes_TP = Bayes_Confusion_matrix(1,1);
Bayes_FN = Bayes_Confusion_matrix(1,2);
Bayes_FP = Bayes_Confusion_matrix(2,1);
Bayes_TN = Bayes_Confusion_matrix(2,2);

RF_TP = RF_Confusion_matrix(1,1);
RF_FN = RF_Confusion_matrix(1,2);
RF_FP = RF_Confusion_matrix(2,1);
RF_TN = RF_Confusion_matrix(2,2);

%Calculate the metrics of the confusion matirx of Naive Bayes model 
Accuracy_Bayes    = (Bayes_TP+Bayes_TN)/(Bayes_TP+Bayes_TN+Bayes_FP+Bayes_FN);
Precision_Bayes   = (Bayes_TP)/(Bayes_TP+Bayes_FP);
Recall_Bayes      = (Bayes_TP)/(Bayes_TP+Bayes_FN);
Specificity_Bayes = (Bayes_TN)/(Bayes_TN+Bayes_FP);
F1_Bayes          = (2*Precision_Bayes*Recall_Bayes)/(Precision_Bayes+Recall_Bayes);
Metric_Bayes      = [Accuracy_Bayes,Precision_Bayes,Recall_Bayes,Specificity_Bayes,F1_Bayes];

%Calculate the metrics of the confusion matirx of Random Forest model 
Accuracy_RF    = (RF_TP+RF_TN)/(RF_TP+RF_FP+RF_TN+RF_FN);
Precision_RF   = (RF_TP)/(RF_TP+RF_FP);
Recall_RF      = (RF_TP)/(RF_TP+RF_FN);
Specificity_RF = (RF_TN)/(RF_TN+RF_FP);
F1_RF          = (2*Precision_RF*Recall_RF)/(Precision_RF+Recall_RF);
Metric_RF      = [Accuracy_RF,Precision_RF,Recall_RF,Specificity_RF,F1_RF];

%Store the results in a table 
metric  = [Metric_Bayes; Metric_RF];
Metric  = array2table(metric);
model    = {'NB','RF'}';
Model   = cell2table(model);
Metric  = [Model Metric];
Metric.Properties.VariableNames = {'Model','Accuracy','Precision','Recall','Specificity','F1'};

%Plot the metrics of both models for comparison  
figure('Name','Metrics for NB and RF')
x = categorical({'Accuracy','Precision','Recall','Specificity','F1'})
y = [Metric_Bayes;Metric_RF]
bar(x,y)
ylim([0.8 1])
legend('NB','RF','Location','northwest')
title('Metrics for NB and RF','FontSize',24)



   