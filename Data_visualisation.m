clear all; clc; close all;

run('Data_preparation.m')

%Plot the normplot for each features 
figure('Name','Normplot')

for i = 1:30
    subplot(6,6,i)
    normplot(X(:,i))
    title(X_table.Properties.VariableNames(i))
end

%Plot the normplot for two extreme featres for comparison 
figure 
normplot(X(:,28))
title(X_table.Properties.VariableNames(28),'FontSize',24)

figure 
normplot(X(:,14))
title(X_table.Properties.VariableNames(14),'FontSize',24)


%Plot the covariance heatmap
cov_matrix = corrcoef(X);

figure('Name','Covariance Heatmap')
x=size(X,2);
imagesc(cov_matrix);
set(gca,'XTick',1:x);
set(gca,'YTick',1:x);
set(gca,'XTickLabel',X_table.Properties.VariableNames,'FontSize',15);
set(gca,'YTickLabel',X_table.Properties.VariableNames,'FontSize',15);
axis([0 x+1 0 x+1]);
xtickangle(45);
title('Covariance Heatmap','Fontsize',24)
grid;
colorbar;

%Separate the class column from table
diagnosis_class = file(:,1);

%Transform data from table to array
diagnosis_class = table2array(diagnosis_class);

%Separate class for histogram
class_labels = categorical(diagnosis_class(:,1));
tabulate(class_labels)

%Transform from cell to matrix and classifying in M and B
T2      = cell2mat(file{:,1})
class_M = file(T2=='M',:);
class_B = file(T2=='B',:);

%Calculate mean for different classes
radius_mean_M  = mean(class_M.radius_mean,1)

texture_mean_M = mean(class_M.texture_mean,1)

radius_mean_B  = mean(class_B.radius_mean,1)

texture_mean_B = mean(class_B.texture_mean,1)

area_mean_B    = mean(class_B.area_mean,1)

area_mean_M    = mean(class_M.area_mean,1)

%Calculate std deviation for different classes
radius_std_M  = std(class_M.radius_mean,1)

texture_std_M = std(class_M.texture_mean,1)

radius_std_B  = std(class_B.radius_mean,1)

texture_std_B = std(class_B.texture_mean,1)

area_std_B    = std(class_B.area_mean,1)

area_std_B    = std(class_B.area_mean,1)

%Calculate the skewness of different classes
radius_skew_M  = skewness(class_M.radius_mean,1)

texture_skew_M = skewness(class_M.texture_mean,1)

radius_skew_B  = skewness(class_B.radius_mean,1)

texture_skew_B = skewness(class_B.texture_mean,1)

area_skew_B    = skewness(class_B.area_mean,1)

area_skew_B    = skewness(class_B.area_mean,1)

%Plot histogram Mean Radius
figure('Name','Histogram of Mean Radius')
h1 = histogram(class_M.radius_mean, 'EdgeAlpha',0.5, 'EdgeColor','b', 'FaceColor','b' )
hold on
h2 = histogram(class_B.radius_mean,  'EdgeAlpha',0.5, 'EdgeColor','r', 'FaceColor','r' )
legend('Malignant','Benign')
title ('Breast Cancer')
xlabel('Mean Radius of Cell Nuclei')
ylabel('Count')

%Plot histogram of mean texture
figure('Name','Histogram of Mean Texture')
h3 = histogram(class_M.texture_mean, 'EdgeAlpha',0.5, 'EdgeColor','b', 'FaceColor','b' )
hold on
h4 = histogram(class_B.texture_mean, 'EdgeAlpha',0.5, 'EdgeColor','r', 'FaceColor','r')
legend('Malignant','Benign')
title('Breast Cancer')
xlabel('Mean Texture of Cell Nuclei')
ylabel('Count')

%Plot Histogram of Mean Area
figure('Name','Histogram of Mean Area')
title('Breast Cancer')
h5 = histogram(class_M.area_mean, 'EdgeAlpha',0.5, 'EdgeColor','b', 'FaceColor','b' )
hold on
h6 = histogram(class_B.area_mean, 'EdgeAlpha',0.5, 'EdgeColor','r', 'FaceColor','r' )
legend('Malignant','Benign')
title('Breast Cancer')
xlabel('Mean Area of Cell Nuclei')
ylabel('Count')
