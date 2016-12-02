% Machine Learning Homework Assignment 3
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% 
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
% Loading data for 1 and 6, 1000 per class.

indices = find(labels_train(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_1 = images_train(indices, :);
labels_train_1 = labels_train(indices, :);
images_train_1 = images_train_1(1:1000, :);
labels_train_1 = labels_train_1(1:1000, :);

indices = find(labels_train(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_2 = images_train(indices, :);
labels_train_2 = labels_train(indices, :);
images_train_2 = images_train_2(1:1000, :);
labels_train_2 = labels_train_2(1:1000, :);

images_train_complete = images_train;
labels_train_complete = labels_train;

images_train = [images_train_1 ; images_train_2];
labels_train = [labels_train_1 ; labels_train_2];

% Stores sorted train data for homework part 4
images_train_sorted = [images_train_1 ; images_train_2];
labels_train_sorted = [labels_train_1 ; labels_train_2];

images_train_combined = zeros(size(images_train));

% Randomizing the train data
indices_loc = randperm(size(images_train,1));
for i = 1:2000
  images_train_combined(i,:) = images_train(indices_loc(i) , :);
end
labels_train_combined = labels_train(indices_loc);

images_train = images_train_combined;
labels_train = labels_train_combined;


% Loading test data
indices = find(labels_test(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_1 = images_test(indices, :);
labels_test_1 = labels_test(indices, :);
images_test_1 = images_test_1(1:1000, :);
labels_test_1 = labels_test_1(1:1000, :);

indices = find(labels_test(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_2 = images_test(indices, :);
labels_test_2 = labels_test(indices, :);
images_test_2 = images_test_2(1:958, :);
labels_test_2 = labels_test_2(1:958, :);

images_test = [images_test_1 ; images_test_2];
labels_test = [labels_test_1 ; labels_test_2];

images_test_combined = zeros(size(images_test));

% Randomizing test data
indices_loc = randperm(size(images_test,1));
for i = 1:size(images_test,1)
  images_test_combined(i,:) = images_test(indices_loc(i) , :);
end
labels_test_combined = labels_test(indices_loc);

images_test = images_test_combined;
labels_test = labels_test_combined;


W = zeros(1,size(images_train,2));

% Homework 3 , Solutions 3

disp('---------------------------Solution 3---------------------------');
C = [0.002, 0.02, 0.2, 2, 20];
disp('Plotting graph for Homework Solution 3');
for c = C
    [W, accuracy_values] = hw3_train_data(W, images_train, labels_train, images_test, labels_test,6,c,1);
    figure();
    plot(accuracy_values);
    axis([1,size(accuracy_values,2),0,1]);
    str = sprintf('Solution 3: C = %f',c);
    disp(str);
    title(str);
    ylabel('Accuracy');
    xlabel('No. of Iterations');
end

W = zeros(1,size(images_train,2));

% Homework 3 , Solutions 4

disp('---------------------------Solution 4---------------------------');
[W, accuracy_values] = hw3_train_data(W, images_train_sorted, labels_train_sorted, images_test, labels_test,6,0.2,1);
figure();

disp('Plotting graph for Homework Solution 4');
plot(accuracy_values);
axis([1,size(accuracy_values,2),0,1]);
title('Solution 4: Accuracy vs Iterations, C = 0.2');
ylabel('Accuracy');
xlabel('No. of Iterations');

clear
load MNIST_digit_data;

rand('seed', 1);

disp('---------------------------Solution 6---------------------------');
W = zeros(10,size(images_train,2));
test = 0:9;
[confusion_matrix, average_accuracy, worst_images] = hw3_multiclassifier(W, images_train, labels_train, images_test, labels_test,test);

str = sprintf('Average Accuracy  = %f',average_accuracy);
disp(str);

disp('---------------------------Solution 7---------------------------');
disp('Displaying the worst images and with the ground truth and predicted truth');
figure();

hold on
for img=1:10   
    im = reshape(images_test(worst_images(img,1), :), [28 28]);
    subplot(2,5,img),imshow(im);
    title(strcat('A:',num2str(img-1),',','P:',num2str(worst_images(img,3))));
end 

