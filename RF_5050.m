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

Training   = file(idx(1:n), :);
Testing    = file(idx(n+1:end), :);

%%%Starting grid search - Hyperparameter Optimimzation

%NumTrees - Number of decision trees in the ensemble.
%MinLeafSize - Minimum number of observations per tree leaf.
%NumPredictorsToSample - Number of predictors for random feature selection.
%NumPredictors - Number of features
%Training 70% of dataset
%Testing 30% of dataset

%Input
%Array with values of number of trees for the forest 
numTree = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200] ; 

%Array of values to try for minimum number of observations per tree leaf
leaf = [1:10]; 

%Array of values to vary the number of predictors to select per split 
numPredictors = [1 : 15] ; 

%Output
%Empty arrays to hold values created in loops 
Parameters = [];
Errors = [] ;  
Final_result = table ; 
Accuracy = [] ; 
Totalerror_Table = [] ; 

%Create list of class labels to use for confusion matrix 
labels_training = table2cell(Training(:,1)); 
labels_testing = Testing{:, 1} ; 

tic
%Validation
for i = 1:length(numTree)
    
    for j = 1:length(leaf)
    
        for k = 1:length(numPredictors)
        
            Mdl= TreeBagger(numTree(i), Training, 'diagnosis', 'OOBPrediction', 'on', 'minLeafSize', leaf(j), ...
                'NumPredictorsToSample', numPredictors(k)); 
            
            %Input the parameters tested into an array
            Parameters = [Parameters; numTree(i), leaf(j), numPredictors(k)]; %input the parameters tested into an array
            
            %Use ensemble to calculate an average for all the trees 
            model_error = oobError(Mdl, 'Mode', 'Ensemble') ; % Out-of-bag error
           
            %Generate the out-of-bag errors for each of the trees in the model for plotting
           Eachtree_Error = oobError(Mdl); 

            %Input the oobError values into an array
            Errors = [Errors; model_error];

            Totalerror_Table = [Totalerror_Table ; Eachtree_Error]; %generate table including error values for each tree

           %Use the trained model to predict classes on the out of bag(OOB)
           %oob scores estimate generalization accuracy
           %observations stored in the model
           [predicted_labels, scores, stdevs]= oobPredict(Mdl);

           %Create the confusion matrix from the ooBPredict - Ensemble predictions for out-of-bag observations. 
           CM_model= confusionmat(labels_training,predicted_labels);

           %Calculate the accuracy of the model using the confusion matrix 
           Validation_accuracy = 100*sum(diag(CM_model))./sum(CM_model(:));

           %Store the model accuracy values in an array
           Accuracy = [Accuracy; Validation_accuracy] ; 

           %Join the parameters test and the model error and accuracy in a row
           %in an array 
           Final_result = [Parameters Errors Accuracy] ; 
        end
    end 
end 
toc

view(Mdl.Trees{1},'Mode', 'graph')

%Plot the out of bag classification error for the number of trees grown
figure;
plot (Eachtree_Error)
xlabel ('Number of grown trees');
ylabel ('Out-of-bag classification error');


%Transform the array  to a table
Final_result = array2table(Final_result) ; 
%Assign column names to table 
Final_result.Properties.VariableNames = {'NumTrees', 'NumLeaves' , 'NumSamples', 'oobErrorValue', 'AccuracyValue'} ; 

%Find the minimum model error
min_error = min(Final_result{:,4}); 

%Find the highest validation accuracy 
highest_Val_Accuracy = max(Final_result{:,5});

%Find the best validation model with the highest validation accuracy and check parameters 
best_val_model = Final_result(Final_result.AccuracyValue == highest_Val_Accuracy, :);

%Print best validation model
head(best_val_model)

for i=1:size(best_val_model)
    
    %Train model using the best parameters from grid search (validation)
    numtrees(i)   = best_val_model{i,1}; 
    numLeaves(i)  = best_val_model{i,2};
    numSamples(i) = best_val_model{i,3}; 

    %Validated Model to use with testing
    Validated_training_mdl = TreeBagger(numtrees(i), Training, 'diagnosis',... 
    'OOBPrediction', 'on', 'minLeafSize', numLeaves(i), 'NumPredictorsToSample', numSamples(i)); 

    %Test the model using predict and the testing dataset 
    [predicted_labels,scores]= predict(Validated_training_mdl, Testing(:, 2:31));

    %Generate a confusion matrix 
    CM_testing_mdl= confusionmat(labels_testing,predicted_labels);

    %Calculating classification accuracy using formula Acc=TP +TN/ TP+TN+FP+FN
    Classification_accuracy = 100*sum(diag(CM_testing_mdl))./sum(CM_testing_mdl(:)) ; 
    Error_Final =  oobError(Validated_training_mdl, 'Mode', 'Ensemble') ;  %average error

end 

    
    



