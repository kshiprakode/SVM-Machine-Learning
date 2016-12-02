% Machine Learning Homework Assignment 3
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% Function to train data

function [confusion_matrix, average_accuracy, worst_images] = hw3_multiclassifier(W, images_train, labels_train, images_test, labels_test,test)
    C = 0.2;
    b = 0;
    W_prime = zeros(1,size(images_train,2));
    for test_data = test
         [W_prime, ~] = hw3_train_data(W_prime, images_train, labels_train, images_test, labels_test,test_data,C,0);
         W(test_data+1,:) = W_prime;
    end
    
    worst_images = zeros(10,3);
    confusion_matrix = zeros(10,10);
    
    for i = 1:size(images_test,1)
        predict = images_test(i,:) * W';
        [sorted,index] = sort(predict,'descend');
    
        confusion_matrix(labels_test(i)+1,index(1)) = confusion_matrix(labels_test(i) + 1,index(1)) + 1;
    
        if ((index(1)-1) ~= labels_test(i))
            if(worst_images(labels_test(i) + 1,2) > sorted(1))
                worst_images(labels_test(i) + 1,1) = i;
                worst_images(labels_test(i) + 1,2) = sorted(1);
                worst_images(labels_test(i) + 1,3) = index(1)-1;
            end
        end

    end
    confusion_matrix = normr(confusion_matrix);
    average_accuracy = trace(confusion_matrix)/10;  
    
end