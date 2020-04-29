rng(1)

%import data and make training and testing sets 
run('Data_preparation.m') 

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
plot   (Eachtree_Error)
xlabel ('Number of grown trees')
ylabel ('Out-of-bag classification error')
title  ('oob Error for the Number of Trees Grown','FontSize',24)

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

%Plot the best_model tree
view(Mdl.Trees{best_val_model},'Mode', 'graph')

%Train model using the best parameters from grid search (validation)
numtrees = best_val_model{:,1} ; 
numLeaves = best_val_model{:,2};
numSamples = best_val_model{:,3} ; 

%Validated Model to use with testing
Validated_training_mdl = TreeBagger(numtrees, Training, 'diagnosis', 'OOBPrediction', 'on', 'minLeafSize', numLeaves, 'NumPredictorsToSample', numSamples); 

%Test the model using predict and the testing dataset 
[predicted_labels,scores]= predict(Validated_training_mdl, Testing(:, 2:31));

%Generate a confusion matrix 
CM_testing_mdl= confusionmat(labels_testing,predicted_labels, 'Order', {'M', 'B'});
figure
confusionchart(CM_testing_mdl)
       
%Calculating classification accuracy using formula Acc=TP +TN/ TP+TN+FP+FN
Classification_accuracy = 100*sum(diag(CM_testing_mdl))./sum(CM_testing_mdl(:)) ; 
Error_Final =  oobError(Validated_training_mdl, 'Mode', 'Ensemble') ;  %average error

