Train_image_vec = uint8(zeros(10304,340));  % Dimension of an image is = 112 x 92 = 10304 pixels; I want to load 40 x 8 [index: 1 to 8] = 340 images for training 
Test_image_vec =  uint8(zeros(10304,60));   
    for i=1:40 
       path = strcat('s',num2str(i),'\');
       if i <=20
         for j=1:8
           img = imread(strcat(path,num2str(j),'.pgm'));
           Train_image_vec (:,j+(i-1)*8) =reshape(img, 10304,1);
         end
         for j=9:10
           img = imread(strcat(path,num2str(j),'.pgm'));
           Test_image_vec (:,(j-8)+(i-1)*2) = reshape(img, 10304,1);
         end
       end
       if i > 20
         for j=1:9
           img = imread(strcat(path,num2str(j),'.pgm'));
           Train_image_vec (:,j+(i)*8) =reshape(img, 10304,1);
         end
         for j=10
           img = imread(strcat(path,num2str(j),'.pgm'));
           Test_image_vec (:,(j-8)+(i-1)*2) = reshape(img, 10304,1);
         end
       end   
    end

zero_column_indices = find(all(Test_image_vec == 0, 1));
Test_image_vec(:,zero_column_indices)=[];




%%2 & 3.
avg=uint8(mean(Train_image_vec,2)); 
Normal_Train_Vec = zeros(size(Train_image_vec));
 for i=1:340
     Normal_Train_Vec (:, i) = Train_image_vec(:, i) - avg;
 End
M = Normal_Train_Vec' * Normal_Train_Vec;
[U, E, V] =svd(M);  

acc_list = zeros(1,340); 
for iter = 1:size(U, 2)
    U_temp = U(:, 1:iter);
    
    Training_Feature = uint8(single(U_temp).*single(Normal_Train_Vec'));

    i = randi(40, 1, 1);
    j = randi(2, 1, 1);

    Test_img = Test_image_vec(:, j + (i - 1) * 2);

    figure;
    subplot(121);
    imshow(reshape(Test_img, 112, 92));
    title('Test image ...');

    Normal_Test_img = uint8(Test_img - avg);
    Test_Feature = uint8(single(Normal_Test_img)' * single(U_temp));

    Match_score = zeros(1,340);
    for k = 1:340
        Match_score = [Match_score, norm(single(Training_Feature(k, :)) - single(Test_Feature), 2)];
    end

    accuracy = 100 - (100 * min(Match_score) / max(Match_score));
    acc_list(iter)= accuracy;
    disp(['Accuracy for iteration ', num2str(iter), ': ', num2str(accuracy), '%']);

    
    [minScore, location] = min(Match_score);
    subplot(122);
    for i = 1:location
        imshow(reshape(Train_image_vec(:, i), 112, 92));
        title(strcat('Searching ..., score= ', num2str(round(Match_score(i)))));
        drawnow;
    end

    title(strcat('Match found!...score =', num2str(round(minScore))));
    wait4bp = waitforbuttonpress;
    close all;
end
