% @Hoque: Face recognition using PCA 
% Note: I used 15 out of 320 Eigen vectors ... sometimes it can produce inaccurate results ... you can optimize threshold for similarity ... you may want to apply more (or less) EigenVectors and then can recheck the accuracy 

% Load the training and test images ==============================================================================
 Train_image_vec = uint8(zeros(10304,320));  % Dimension of an image is = 112 x 92 = 10304 pixels; I want to load 40 x 8 [index: 1 to 8] = 320 images for training
  Test_image_vec =  uint8(zeros(10304,80));   % I want to load 40 x 2 [index: 9 to 10] = 80 images for testing
    for i=1:40 
       path = strcat('s',num2str(i),'\');
       for j=1:8
                img = imread(strcat(path,num2str(j),'.pgm'));
                Train_image_vec (:,j+(i-1)*8) =reshape(img, 10304,1);
       end
       for j=9:10
                img = imread(strcat(path,num2str(j),'.pgm'));
                Test_image_vec (:,(j-8)+(i-1)*2) = reshape(img, 10304,1);
       end     
    end
%=================================================================================================================   
% Take the average of the respective pixel-values of all the training images =====================================
avg=uint8(mean(Train_image_vec,2));  % 2=> row-wise avg

% [Optional Step:] See how the average-image looks like ==========================================================
  imshow(reshape(avg,112,92));  title('This is the average of the training faces');   
  wait4bp = waitforbuttonpress;

% Remove the computed average value from the each of the training images and  create "Normal vector"==============
 Normal_Train_Vec =[];
 for i=1:320
     Normal_Train_Vec (:, i) = Train_image_vec(:, i) - avg;
 end

%Generate covariance matrix M ====================================================================================
M = Normal_Train_Vec' * Normal_Train_Vec;
[U, E, V] =svd(M);     % U = EigenVector, E = EigenValue

%Let us pick 1st best 15 EigenVectors' corresponsing normal training vectors (from 10304x320 to 10304x15) ========
U=Normal_Train_Vec * U; % Project the normal training vectors towards the EigenValues 
U15=U(:,1:15);          % Take 15 Eigen_normal_training_vec 

%[Optional Steps:] Show the 1st 15 Eigen Faces (using U15) =======================================================
  U15_faces = [];
  figure;
  for i=1:15
         U15_faces =reshape (U15(:,i),112,92);  %+reshape(avg,112,92); 
         subplot(6,5,i); % display Eigen Faces in 5 x 3 pattern 
         imshow(U15_faces, []);
         title(strcat('Eigen Face#',num2str(i)));
  end     	
  wait4bp = waitforbuttonpress

% Extract the features from training ... each row in Training_Feature is the pattern of one training images.======
   Training_Feature =[];
   for i =1:320
       Training_Feature(i,:)=  uint8(single(Normal_Train_Vec(:,i))'*single(U15));   
   end 

% I want to randomly pick one image from the Test_image_vec [size= (40 x 2)], to test recognition ================
  i= randi(40,1,1); 
  j= randi(2,1,1);

Test_img=Test_image_vec(:,j+(i-1)*2);

figure;
subplot(121); 
imshow(reshape(Test_img,112,92));
title('Test image ...');

Normal_Test_img=uint8(Test_img-avg); % remove training average
Test_Feature  =  uint8(single(Normal_Test_img)'*single(U15));                            


Match_score=[];
 for k=1:320
     Match_score=[Match_score,norm(single(Training_Feature(k,:))-single(Test_Feature),2)];
 end

[score,location]=min(Match_score);

wait4bp = waitforbuttonpress;

subplot(122);

for i=1:location
    imshow(reshape(Train_image_vec(:,i),112,92));title(strcat('Searching ..., score= ',num2str(round(Match_score(i)))));drawnow;
end      

title(strcat('Match found!...score =',num2str(round(Match_score(i)))));
wait4bp = waitforbuttonpress;
close all;

