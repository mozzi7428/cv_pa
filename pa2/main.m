%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the skeleton code of PA2 in EC5301 Computer Vision.              %
% It will help you to implement the Structure-from-Motion method easily.   %
% Using this skeleton is recommended, but it's not necessary.              %
% You can freely modify it or you can implement your own program.          %
% If you have a question, please send me an email to haegonj@gist.ac.kr    %
%                                                      Prof. Hae-Gon Jeon  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;

addpath('Givenfunctions');

%% Define constants and parameters
% Constants ( need to be set )
num_trials    = 10000;

% Thresholds ( need to be set )
threshold_of_distance = 4.0e-6; 

edge_thresh = 7;
peak_thresh = 3;
match_thresh = 3.0;

% Matrices
K               = [ 1698.873755 0.000000     971.7497705;
                    0.000000    1698.8796645 647.7488275;
                    0.000000    0.000000     1.000000 ];

inlier_thres = 1.0e-5;
%% Feature extraction and matching
% Load images and extract features and find correspondences.
% Fill num_Feature, Feature, Descriptor, num_Match and Match
% hints : use vl_sift to extract features and get the descriptors.
%        use vl_ubcmatch to find corresponding matches between two feature sets.

% 1. Load the input images (‘sfm01.JPG’, ‘sfm02.JPG’)
img1 = imread('data_2_sfm/sfm01.JPG');
img2 = imread('data_2_sfm/sfm02.JPG');

gray1 = single(rgb2gray(img1));
gray2 = single(rgb2gray(img2));


% 2. Extract features from both images using the function ‘vl_sift’ (2 pts)
[f1, d1] = vl_sift(gray1, 'EdgeThresh', edge_thresh, 'PeakThresh', peak_thresh);
[f2, d2] = vl_sift(gray2, 'EdgeThresh', edge_thresh, 'PeakThresh', peak_thresh);


% 3. Match features (find correspondence) between two images using the function ‘vl_ubcmatch’ (2 pts)
[matches, scores] = vl_ubcmatch(d1, d2, match_thresh) ;

% collect X's
X1 = f1([2,1], matches(1, :));
X2 = f2([2,1], matches(2, :));

% X_hat = inverse(K) * X
X1_hat = [X1; ones(1, size(X1,2))];
X2_hat = [X2; ones(1, size(X2,2))];
X1_hat = inv(K) * X1_hat;
X2_hat = inv(K) * X2_hat;

% %% Plot SIFT
% figure(1) ; clf ;
% imagesc(cat(2, img1, img2)) ;
% 
% xa = f1(1,matches(1,:)) ;
% xb = f2(1,matches(2,:)) + size(img1,2) ;
% ya = f1(2,matches(1,:)) ;
% yb = f2(2,matches(2,:)) ;
% 
% hold on ;
% h = line([xa ; xb], [ya ; yb]) ;
% set(h,'linewidth', 1, 'color', 'b') ;
% 
% vl_plotframe(f1(:,matches(1,:))) ;
% f2(1,:) = f2(1,:) + size(img1,2) ;
% vl_plotframe(f2(:,matches(2,:))) ;
% axis image off ;

%% InitI1lization step
% 4. Estimate EssentI1l matrix E with RANSAC using ‘calI2rated_fivepoint’ (8pts)

max_inliers = 0;
best_mse = nan;
% choose best supportive hypothesis E having the most inliers
for i=1:num_trials
    % (1) Randomly select sets of 5 points
    random_idx = randsample(size(matches, 2), 5);
    x1 = X1_hat(:, random_idx);
    x2 = X2_hat(:, random_idx);
    
    % apply calibrated-fivepoint function
    Evec = calibrated_fivepoint(x1, x2);
    
    % (2) Generate E(hypothesis) and evaluate using other points with pre-defined threshold - epipolar distance
    for j=1:size(Evec, 2)
        E = reshape(Evec(:,j),3,3);
        
        inliers = zeros(size(matches, 2), 1);
        cnt = 1;
        mse = 0;
        for k = 1 : size(matches, 2)
           distance = abs(X1_hat(:, k)' * E * X2_hat(:,k));
           if distance > threshold_of_distance
               inliers(cnt) = k;
               cnt = cnt+1;
               mse = mse + distance^2;
           end
        end
        
        num_inliers = cnt-1;
        if num_inliers > max_inliers
           best_E = E;
           best_inliers = inliers(1:cnt-1);
           max_inliers = num_inliers;
           best_mse = mse;
        elseif num_inliers == max_inliers && mse < best_mse
                best_E = E;
                best_inliers = inliers(1:cnt-1);
                max_inliers = num_inliers;
                best_mse = mse;
        end
    end
end
E = best_E; % find out




% Plot SIFT
inmatches = matches(:, best_inliers);
figure(2) ; clf ;
imagesc(cat(2, img1, img2)) ;

xa = f1(1,inmatches(1,:)) ;
xb = f2(1,inmatches(2,:)) + size(img1,2) ;
ya = f1(2,inmatches(1,:)) ;
yb = f2(2,inmatches(2,:)) ;

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'b') ;

vl_plotframe(f1(:,inmatches(1,:))) ;
f2(1,:) = f2(1,:) + size(img1,2) ;
vl_plotframe(f2(:,inmatches(2,:))) ;
axis image off ;










% 5. Decompose essentI1l matrix E to camera extrinsic [R|T] (6 pts)
[U, S, V] = svd(E);
u3 = U(:,3);
W = [   [0, -1, 0]; 
        [1, 0, 0];
        [0, 0, 1]];

% possible cams
possibles = {   [U * W * V', u3];
                [U * W * V', -u3]; 
                [U * W' * V', u3];
                [U * W' * V', -u3]};
        
x1 = X1_hat(:, best_inliers);
x2 = X2_hat(:, best_inliers);
% x1 = x1(:,1);
% x2 = x2(:,1);

num_correct = 0;
cam = possibles{1};
for i = 1:4
    P2 = possibles{i};
    X = triangulation_jy(P2, x1(:, i), x2(:, i));

    temp = dot((X(1:3) - P2 * [0;0;0;1]), P2 * [0;0;1;0]);
    if X(3)>0 && temp>0
       cam = P2;
    end
end

R = cam(:, 1:3); % find out
T = cam(:, 4); % find out


% 6. Generate 3D points by implementing TrI1ngulation (7 pts)
X = zeros(4, size(x1, 2));
for i=1:size(x1, 2)
    X(:, i) = triangulation_jy(cam, x1(:,i), x2(:,i));
end;
x1 = X1(:, best_inliers);
x2 = X2(:, best_inliers);

X_with_color = []; % find out
for i=1:size(x1,2)
    color = img1(round(x1(1, i)), round(x1(2, i)), :);
    color = double(reshape(color,3,1)) / 255;
    X_with_color = [X_with_color [X(1:3, i); color]];
end

% Save 3D points to PLY
SavePLY('2_views.ply', X_with_color);
