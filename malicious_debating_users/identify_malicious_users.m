
%%%%%%%%%%%%%%%%%%%%%%
%%% INITIALIZATION %%%
%%%%%%%%%%%%%%%%%%%%%%

clear ; close all; clc

fprintf('Loading the training set... \n \n');

% Load all the data from the file "training_set_userBehavior.txt" into the matrix "training_set". 
training_set = load('training_set_userBehavior.txt');

% We want the matrix "X_train" to contain only information about the "observable" debate features.
% Each row of the matrix "X_train" contains information on the behavior of one agent, during a single debate.
% column 1: activity
% column 2: (activity - avg activity)
% column 3: opinionatedness 
% column 4: (opinionatedness - avg opinionatedness)
% column 5: classifiability 
% column 6: (classifiability - avg classifiability) 
X_train = training_set(:,1:6);
%%X_train_a = training_set(:,2);
%%X_train_b = training_set(:,4);
%%X_train_c = training_set(:,6);
%%X_train = [X_train_a X_train_b X_train_c];

% Some useful values:
% m: (number of agents in a debate) x (number of debates)
% n: number of "observable" debate features.
m = rows(X_train);
n = columns(X_train);

% Add a column of ones at the beginning of table "X_train".
% Useful for the regression we'll do shortly.
X_train = [ones(m,1) X_train];

% Firstly, we try to predict if an agent has lied during a debate.
% This information is found at the 8th column of the table "training_set".
y_hasLied_train = training_set(:,8);

% Secondly, we try to predict if an agent has hidden (maliciously) during a debate.
% This information is found at the 12th column of the table "training_set".
y_hasHidden_train = training_set(:,12);

% In order to learn how to predict if an agent has lied or hidden during a debate, we'll use logistic regression. 
% We define the parameter vectors "init_theta_hasLied" and "init_theta_hasHidden".
% Both vectors have (n+1) rows.
init_theta_hasLied = zeros(n+1,1);
init_theta_hasHidden = zeros(n+1,1);


%%%%%%%%%%%%%%%%
%%% TRAINING %%%
%%%%%%%%%%%%%%%%

fprintf('Running logistic regression... \n \n');

% Parameters of the logistic regression.
alpha = 0.05;
num_iterations = 3000;
lambda = 0;
options = optimset('GradObj', 'on', 'MaxIter', num_iterations);

% Compute the (optimal) vector "theta_hasLied" using the function "fminunc()".
[theta_hasLied, J_hasLied, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_train, y_hasLied_train, lambda)), init_theta_hasLied, options);
  
% Compute the optimal vector "theta_hasHidden" using the function "fminunc()".
[theta_hasHidden, J_hasHidden, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_train, y_hasHidden_train, lambda)), init_theta_hasHidden, options);

fprintf('Vector theta_hasLied (computed by fminunc): \n');
fprintf(' %f \n', theta_hasLied);
fprintf('\n');
fprintf('Vector theta_hasHidden (computed by fminunc): \n');
fprintf(' %f \n', theta_hasHidden);
 fprintf('\n');
 

%%%%%%%%%%%%%%%%%%%
%%% PREDICTIONS %%%
%%%%%%%%%%%%%%%%%%%

% Load all the data from the file "test_set_userBehavior.txt" into the matrix "test_set". 
test_set = load('test_set_userBehavior.txt');

% We want the matrix "X_test" to contain only information about the "observable" debate features.
% Therefore, similarly to what we did for the table "X_train":
X_test = test_set(:,1:6);
%%X_test_a = test_set(:,2);
%%X_test_b = test_set(:,4);
%%X_test_c = test_set(:,6);
%%X_test = [X_test_a X_test_b X_test_c];

X_test = [ones(rows(X_test),1) X_test];

y_hasLied_test = test_set(:,8);
y_hasHidden_test = test_set(:,12);

predict_liars = predict(theta_hasLied, X_test);
predict_hiders = predict(theta_hasHidden, X_test);

fprintf('Results (LIARS): \n');
fprintf('**************** \n');
tp_liars = sum((predict_liars==1) & (y_hasLied_test==1));
fp_liars = sum((predict_liars==1) & (y_hasLied_test==0));
tn_liars = sum((predict_liars==0) & (y_hasLied_test==0));
fn_liars = sum((predict_liars==0) & (y_hasLied_test==1));
precision_liars = tp_liars / (tp_liars + fp_liars);
recall_liars    = tp_liars / (tp_liars + fn_liars);
accuracy_liars  = (tp_liars + tn_liars) / (tp_liars + fp_liars + tn_liars + fn_liars);
F1_score_liars  = (2 * precision_liars * recall_liars) / (precision_liars + recall_liars);
fprintf('True positives: %d \n', tp_liars);
fprintf('False positives: %d \n', fp_liars);
fprintf('True negatives: %d \n', tn_liars);
fprintf('False negatives: %d \n', fn_liars);
fprintf('Precision: %f \n', precision_liars);
fprintf('Recall: %f \n', recall_liars);
fprintf('Accuracy: %f \n', accuracy_liars);
%fprintf('Accuracy: %f \n', mean(double(predict_liars == y_hasLied_test)) * 100);
fprintf('F1-score: %f \n \n', F1_score_liars);

fprintf('Results (HIDERS): \n');
fprintf('***************** \n');
tp_hiders = sum((predict_hiders==1) & (y_hasHidden_test==1));
fp_hiders = sum((predict_hiders==1) & (y_hasHidden_test==0));
tn_hiders = sum((predict_hiders==0) & (y_hasHidden_test==0));
fn_hiders = sum((predict_hiders==0) & (y_hasHidden_test==1));
precision_hiders = tp_hiders / (tp_hiders + fp_hiders);
recall_hiders    = tp_hiders / (tp_hiders + fn_hiders);
accuracy_hiders  = (tp_hiders + tn_hiders) / (tp_hiders + fp_hiders + tn_hiders + fn_hiders);
F1_score_hiders  = (2 * precision_hiders * recall_hiders) / (precision_hiders + recall_hiders);
fprintf('True positives: %d \n', tp_hiders);
fprintf('False positives: %d \n', fp_hiders);
fprintf('True negatives: %d \n', tn_hiders);
fprintf('False negatives: %d \n', fn_hiders);
fprintf('Precision: %f \n', precision_hiders);
fprintf('Recall: %f \n', recall_hiders);
fprintf('Accuracy: %f \n', accuracy_hiders);
%fprintf('Accuracy: %f \n', mean(double(predict_hiders == y_hasHidden_test)) * 100);
fprintf('F1-score: %f \n \n', F1_score_hiders);


%%% Try testing the predictions on the training set as well...
%predict_liars = predict(theta_hasLied, X_train);
%predict_hiders = predict(theta_hasHidden, X_train);
%fprintf('Accuracy of finding liars in train set: %f \n', mean(double(predict_liars == y_hasLied_train)) * 100);
%fprintf('Accuracy of finding hiders in train set: %f \n', mean(double(predict_hiders == y_hasHidden_train)) * 100);


%%% SOLUTION WITH LINEAR REGRESSION - PREDICT THE NUMBER OF LIES... (BEGIN)
% Firstly, we focus on the number of lies made (each time) by one agent, during a single debate.
% This information is found at the 9th column of the table "training_set".
%y_numLies = training_set(:,9);
%% Init Theta and Run Gradient Descent 
%[theta, J_history] = gradientDescentMulti(X, y_numLies, theta, alpha, num_iters);
%% Plot the convergence graph
%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');
%% Display gradient descent's result
%fprintf('Theta computed from gradient descent: \n');
%fprintf(' %f \n', theta);
%fprintf('\n');
%predictions = round(predictions);
%succ_predictions = (predictions == y_numLies2);
%fprintf('We have made %d successful predictions, out of %d. \n', sum(succ_predictions), length(predictions));
%%% SOLUTION WITH LINEAR REGRESSION (END)
