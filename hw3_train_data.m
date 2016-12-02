% Machine Learning Homework Assignment 3
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% Function to train data

function [W, accuracy_vector] = hw3_train_data(W, images_train, labels_train, images_test, labels_test,test, C ,acc_boolean)
    b = 0;
    accuracy_vector = zeros(1,size(images_train,2));
    for t = 1:1
        for i = (1:size(images_train,1))
            learning_rate = 1/i;
            
            label = labels_train(i);
            if(acc_boolean == 1)
                if(label == test)
                    label = -1;
                end
            else
                if(label ~= test)
                    label = -1;
                else
                    label = 1;
                end
            end
            value = images_train(i,:) * W' * label + b;
            
            
            if(value < 1)
                   W = W - learning_rate * ( W - C * label * images_train(i,:));
                   b = b - learning_rate * ( b - C * label);
            else
                W = W - learning_rate * W;
                b = b - learning_rate * b;
            end
            
            if acc_boolean == 1
                acc = hw3_test_data(images_test, labels_test, W, test);
                accuracy_vector(i) = acc;
            end
        end
    end 
end