clear all; close all; clc

% Test 1.

cam1_1 = load('cam1_1.mat');
cam1_1 = cam1_1.vidFrames1_1; % when loaded it gives a struct data type.
cam2_1 = load('cam2_1.mat');
cam2_1 = cam2_1.vidFrames2_1;
cam3_1 = load('cam3_1.mat');
cam3_1 = cam3_1.vidFrames3_1;

%%

% Cut to the same length.
[m, n, b, numFrames]=size(cam1_1); 
% It's a m by n pic per frame. b is 3 since it's in color.
cam2_1 = cam2_1(:,:,:,1:numFrames);
cam3_1 = cam3_1(:,:,:,1:numFrames);
% We need to cropped them to be the same length as the first one for SVD.

%%

% Viewing the video.
for k = 1 : numFrames
mov(k).cdata = cam1_1(:,:,:,k); mov(k).colormap = [];
end
for j=1:numFrames X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Transform to gray scale to save computing later.
cam1_1bw = zeros(m, n, numFrames); 
% Gray scale only has one number for each pixel.
cam2_1bw = zeros(m, n, numFrames);
cam3_1bw = zeros(m, n, numFrames);

for k = 1:numFrames
    
    cam1_1bw(:,:,k) = rgb2gray(cam1_1(:,:,:,k));
    cam2_1bw(:,:,k) = rgb2gray(cam2_1(:,:,:,k));
    cam3_1bw(:,:,k) = rgb2gray(cam3_1(:,:,:,k));
    
end

%%

% Change to uint8 data type, since we got double datatype from last section
% and uint8 is the typical datatype expected for the following code.
cam1_1bw = uint8(cam1_1bw);
cam2_1bw = uint8(cam2_1bw);
cam3_1bw = uint8(cam3_1bw);

%%

% Viewing gray movie.
for k = 1 : numFrames
    mov(k).cdata = cam3_1bw(:,:,k); mov(k).colormap = gray;
end
for j=1:numFrames 
    X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Find x_a, y_a dataset.

% Find the starting point to track along the frames.
I = cam1_1bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

% Initialize multiple starting points for tracking and pick the best one
% later.
point_pos_1 = [333 261];
point_pos_2 = [333 268];
point_pos_3 = [320 223];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam1_1bw(:,:,1));

for i = 2:numFrames % We already know the first position.
    [points, validity]=pointTracker(cam1_1bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam1_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can use point_pos_1 or 2 or 3 as our dataset for x_a, y_a.
% We use point_pos_3 here. 
% We need X_a, Y_a in the matrix to be row vector.
XY_a = transpose(point_pos_3);

%%

% Find x_b, y_b dataset.

I = cam2_1bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [294 316];
point_pos_2 = [294 306];
point_pos_3 = [295 320];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam2_1bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam2_1bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam2_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can use point_pos_1 or 2 or 3 as our dataset for x_b, y_b.
% We use point_pos_3 here. 
XY_b = transpose(point_pos_3);

%%

% Find x_c, y_c dataset.

I = cam3_1bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [314 273];
point_pos_2 = [316 268];
point_pos_3 = [355 320];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam3_1bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam3_1bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam3_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can only use point_pos_3 here as our dataset for x_c, y_c.
XY_c = transpose(point_pos_3);

X = [XY_a; XY_b; XY_c];

% We need to transpose here to get 6x6 result.
CovX = cov(X'); % 6x6.

% As we can see in the covaraince matrix, off diagonal, the covariance
% between Y_a and X_c is very large and very close to the variance of X_c,
% thus we should be aware that X_c might be redundant.

% Y_a, Y_b and X_c has very large variance, thus suggesting the dynamics of
% interests.

%%

[U, S, V] = svd(X);
lambda = diag(S); % Eigenvalues.

figure(1)
% We can see the first mode is completely dominating.
plot(lambda,'ro', 'Linewidth', [1.5]);
xlabel('Mode'), ylabel('Eigenvalue'), title('Visualization of Eigenvalues');

energy1=lambda(1)/sum(lambda);
energy2=lambda(2)/sum(lambda);
energy3=lambda(3)/sum(lambda);

% The first mode is dominating 86.31% of the energy.

xx = U(:,1)*S(1,1)*V(:,1)';
% We can see xx and X are close.

%%

figure(2)
% we also present all five modes here.
for j = 1:5
    xx = U(:,1:j)*S(1:j,1:j)*V(:,1:j)';
    for k = 1:6
        subplot(6,5,j+(k-1)*5)
        plot([1:226], X(k,:), 'r-',[1:226],xx(k,:),'k-');
        set(gca,'Ylim',[100 500]);
        legend('Original X or Y', ['Mode 1:' num2str(j)]);
    end
end

% We can see from the graph, the first three modes can capture most of the
% features of X.

%%
figure(3)
plot([1:6],U(:,1),'k',[1:6],U(:,2),'k--',[1:6],U(:,3),'k:','Linewidth',[2]);
legend('mode 1','mode 2','mode 3');

%%

% Test 2.
% Dealing with white noise: 

clear all; close all; clc

cam1_2 = load('cam1_2.mat');
cam1_2 = cam1_2.vidFrames1_2;
cam2_2 = load('cam2_2.mat');
cam2_2 = cam2_2.vidFrames2_2;
cam3_2 = load('cam3_2.mat');
cam3_2 = cam3_2.vidFrames3_2;

%%

% Cut to the same length.
[m, n, b, numFrames]=size(cam1_2);
cam2_2 = cam2_2(:,:,:,1:numFrames);
cam3_2 = cam3_2(:,:,:,1:numFrames);

%%

for k = 1 : numFrames
mov(k).cdata = cam1_2(:,:,:,k); mov(k).colormap = [];
end
for j=1:numFrames X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Transform to gray scale.
cam1_2bw = zeros(m, n, numFrames);
cam2_2bw = zeros(m, n, numFrames);
cam3_2bw = zeros(m, n, numFrames);

for k = 1:numFrames
    
    cam1_2bw(:,:,k) = rgb2gray(cam1_2(:,:,:,k));
    cam2_2bw(:,:,k) = rgb2gray(cam2_2(:,:,:,k));
    cam3_2bw(:,:,k) = rgb2gray(cam3_2(:,:,:,k));
    
end

%%

% Change to uint8 data type.
cam1_2bw = uint8(cam1_2bw);
cam2_2bw = uint8(cam2_2bw);
cam3_2bw = uint8(cam3_2bw);

%%

for k = 1 : numFrames
    mov(k).cdata = cam2_2bw(:,:,k); mov(k).colormap = gray;
end
for j=1:numFrames 
    X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Find x_a, y_a dataset.

I = cam1_2bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [317 338];
point_pos_2 = [341 345];
point_pos_3 = [341 339];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam1_2bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam1_2bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam1_2(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can only use point_pos_1 as our dataset for x_a, y_a.
XY_a = transpose(point_pos_1);

%%

% Find x_b, y_b dataset.

I = cam2_2bw(:,:,1);
corners= detectFASTFeatures(I); %detectMinEigenFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [382 141];
point_pos_2 = [334 134];
point_pos_3 = [373 223];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam2_2bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam2_2bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam2_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can use point_pos_1 or 2 or 3 as our dataset for x_b, y_b.
% We use point_pos_3 here. 
XY_b = transpose(point_pos_3);

%%

% Find x_c, y_c dataset.

I = cam3_2bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [349 216];
point_pos_2 = [349 224];
point_pos_3 = [375 246];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam3_2bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam3_2bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam3_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can only use point_pos_3 here as our dataset for x_c, y_c.
XY_c = transpose(point_pos_3);

X = [XY_a; XY_b; XY_c];

CovX = cov(X'); % 6x6.

% As we can see in the covaraince matrix, off diagonal, the covariance
% between Y_a and X_c is very large and very close to the variance of X_c,
% thus we should be aware that X_c might be redundant.

% Y_a, Y_b and X_c has very large variance, thus suggesting the dynamics of
% interests.

%%

[U, S, V] = svd(X);
lambda = diag(S); % Eigenvalues.

figure(1)
% We can see the first mode is completely dominating.
plot(lambda,'ro', 'Linewidth', [1.5]);
xlabel('Mode'), ylabel('Eigenvalue'), title('Visualization of Eigenvalues');

energy1=lambda(1)/sum(lambda);
energy2=lambda(2)/sum(lambda);
energy3=lambda(3)/sum(lambda);

% The first mode is dominating 99.05% of the energy.
xx = U(:,1)*S(1,1)*V(:,1)';
% We can see xx and X are close.

%%

figure(2)
% we also present all five modes here.
for j = 1:5
    xx = U(:,1:j)*S(1:j,1:j)*V(:,1:j)';
    for k = 1:6
        subplot(6,5,j+(k-1)*5)
        plot([1:numFrames], X(k,:), 'r-',[1:numFrames],xx(k,:),'k-');
        set(gca,'Ylim',[100 500]);
        legend('Original X or Y', ['Mode 1:' num2str(j)]);
    end
end

% We can see from the graph, the first three modes can capture most of the
% features of X.

%%

figure(3)
plot([1:6],U(:,1),'k',[1:6],U(:,2),'k--',[1:6],U(:,3),'k:','Linewidth',[2]);
legend('mode 1','mode 2','mode 3');

%%

clear all; close all; clc

% Test 3.

cam1_3 = load('cam1_3.mat');
cam1_3 = cam1_3.vidFrames1_3;
cam2_3 = load('cam2_3.mat');
cam2_3 = cam2_3.vidFrames2_3;
cam3_3 = load('cam3_3.mat');
cam3_3 = cam3_3.vidFrames3_3;

%%

% Cut to the same length.
[m, n, b, numFrames]=size(cam3_3);
cam1_3 = cam1_3(:,:,:,1:numFrames);
cam2_3 = cam2_3(:,:,:,1:numFrames);

%%

for k = 1 : numFrames
mov(k).cdata = cam1_3(:,:,:,k); mov(k).colormap = [];
end
for j=1:numFrames X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Transform to gray scale.
cam1_3bw = zeros(m, n, numFrames);
cam2_3bw = zeros(m, n, numFrames);
cam3_3bw = zeros(m, n, numFrames);

for k = 1:numFrames
    
    cam1_3bw(:,:,k) = rgb2gray(cam1_3(:,:,:,k));
    cam2_3bw(:,:,k) = rgb2gray(cam2_3(:,:,:,k));
    cam3_3bw(:,:,k) = rgb2gray(cam3_3(:,:,:,k));
    
end

%%

% Change to uint8 data type.
cam1_3bw = uint8(cam1_3bw);
cam2_3bw = uint8(cam2_3bw);
cam3_3bw = uint8(cam3_3bw);

%%

for k = 1 : numFrames
    mov(k).cdata = cam3_3bw(:,:,k); mov(k).colormap = gray;
end
for j=1:numFrames 
    X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Find x_a, y_a dataset.

I = cam1_3bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [322 284];
point_pos_2 = [351 322];
point_pos_3 = [337 297];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam1_3bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam1_3bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam1_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can use point_pos_1 or 2 or 3 as our dataset for x_a, y_a.
% We use point_pos_1 here. 
XY_a = transpose(point_pos_1);

%%

% Find x_b, y_b dataset.

I = cam2_3bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [268 336];
point_pos_2 = [271 336];
point_pos_3 = [269 336];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam2_3bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam2_3bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam2_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can use point_pos_2 or 3 as our dataset for x_b, y_b.
% We use point_pos_3 here. 
XY_b = transpose(point_pos_3);

%%

% Find x_c, y_c dataset.

I = cam3_3bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [349 225];
point_pos_2 = [382 244];
point_pos_3 = [384 226];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam3_3bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam3_3bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam3_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can only use point_pos_3 here as our dataset for x_c, y_c.
XY_c = transpose(point_pos_3);

X = [XY_a; XY_b; XY_c];

CovX = cov(X'); % 6x6.

% As we can see in the covaraince matrix, off diagonal, the covariance
% between Y_a and X_c is very large and very close to the variance of X_c,
% thus we should be aware that X_c might be redundant.

% Y_a, Y_b and X_c has very large variance, thus suggesting the dynamics of
% interests.

%%

[U, S, V] = svd(X);

lambda = diag(S); % Eigenvalues.

figure(1)
% We can see the first mode is completely dominating.
plot(lambda,'ro', 'Linewidth', [1.5]);
xlabel('Mode'), ylabel('Eigenvalue'), title('Visualization of Eigenvalues');

energy1=lambda(1)/sum(lambda);
energy2=lambda(2)/sum(lambda);
energy3=lambda(3)/sum(lambda);

% The first mode is dominating 99.05% of the energy.
xx = U(:,1)*S(1,1)*V(:,1)';
% We can see xx and X are close.

%%

figure(2)
% we also present all five modes here.
for j = 1:5
    xx = U(:,1:j)*S(1:j,1:j)*V(:,1:j)';
    for k = 1:6
        subplot(6,5,j+(k-1)*5)
        plot([1:numFrames], X(k,:), 'r-',[1:numFrames],xx(k,:),'k-');
        set(gca,'Ylim',[100 500]);
        legend('Original X or Y', ['Mode 1:' num2str(j)]);
    end
end

% We can see from the graph, the first three modes can capture most of the
% features of X.

%%

figure(3)
plot([1:6],U(:,1),'k',[1:6],U(:,2),'k--',[1:6],U(:,3),'k:','Linewidth',[2]);
legend('mode 1','mode 2','mode 3');

%%

clear all; close all; clc

% Test 4.

cam1_4 = load('cam1_4.mat');
cam1_4 = cam1_4.vidFrames1_4;
cam2_4 = load('cam2_4.mat');
cam2_4 = cam2_4.vidFrames2_4;
cam3_4 = load('cam3_4.mat');
cam3_4 = cam3_4.vidFrames3_4;

%%

% Cut to the same length.
[m, n, b, numFrames]=size(cam1_4);
cam2_4 = cam2_4(:,:,:,1:numFrames);
cam3_4 = cam3_4(:,:,:,1:numFrames);

%%

for k = 1 : numFrames
mov(k).cdata = cam1_4(:,:,:,k); mov(k).colormap = [];
end
for j=1:numFrames X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Transform to gray scale.
cam1_4bw = zeros(m, n, numFrames);
cam2_4bw = zeros(m, n, numFrames);
cam3_4bw = zeros(m, n, numFrames);

for k = 1:numFrames
    
    cam1_4bw(:,:,k) = rgb2gray(cam1_4(:,:,:,k));
    cam2_4bw(:,:,k) = rgb2gray(cam2_4(:,:,:,k));
    cam3_4bw(:,:,k) = rgb2gray(cam3_4(:,:,:,k));
    
end

%%

% Change to uint8 data type.
cam1_4bw = uint8(cam1_4bw);
cam2_4bw = uint8(cam2_4bw);
cam3_4bw = uint8(cam3_4bw);

%%

for k = 1 : numFrames
    mov(k).cdata = cam3_4bw(:,:,k); mov(k).colormap = gray;
end
for j=1:numFrames 
    X=frame2im(mov(j)); imshow(X); drawnow
end

%%

% Find x_a, y_a dataset.

I = cam1_4bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [383 268];
point_pos_2 = [414 301];
point_pos_3 = [414 338];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam1_4bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam1_4bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam1_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end
%%
% We can only use point_pos_1 as our dataset for x_a, y_a.

XY_a = transpose(point_pos_1);

%%

% Find x_b, y_b dataset.

I = cam2_4bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [273 160];
point_pos_2 = [290 207];
point_pos_3 = [290 202];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam2_4bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam2_4bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam2_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

%%
% We can use point_pos_2 or 3 as our dataset for x_b, y_b.
% We use point_pos_3 here. 
XY_b = transpose(point_pos_3);

%%

% Find x_c, y_c dataset.

I = cam3_4bw(:,:,1);
corners=detectFASTFeatures(I);
imshow(I); hold on;
plot(corners.selectStrongest(100));

%%

% Restart pointTracker.
pointTracker = vision.PointTracker('MaxBidirectionalError',1);
videoPlayer = vision.VideoPlayer;

point_pos_1 = [361 244];
point_pos_2 = [392 237];
point_pos_3 = [366 256];
initialize(pointTracker, [point_pos_1; point_pos_2; point_pos_3], cam3_4bw(:,:,1));

for i = 2:numFrames
    [points, validity]=pointTracker(cam3_4bw(:,:,i));
    point_pos_1(i,:) = points(1,:);
    point_pos_2(i,:) = points(2,:);
    point_pos_3(i,:) = points(3,:);
    % out=insertMarker(cam3_1(:,:,i), points(validity, :),'+');
    % videoPlayer(out);
end

% We can only use point_pos_3 here as our dataset for x_c, y_c.
XY_c = transpose(point_pos_3);

X = [XY_a; XY_b; XY_c];

CovX = cov(X'); % 6x6.

% As we can see in the covaraince matrix, off diagonal, the covariance
% between Y_a and X_c is very large and very close to the variance of X_c,
% thus we should be aware that X_c might be redundant.

% Y_a, Y_b and X_c has very large variance, thus suggesting the dynamics of
% interests.

%%

[U, S, V] = svd(X);

lambda = diag(S); % Eigenvalues.

figure(1)
% We can see the first mode is completely dominating.
plot(lambda,'ro', 'Linewidth', [1.5]);
xlabel('Mode'), ylabel('Eigenvalue'), title('Visualization of Eigenvalues');

energy1=lambda(1)/sum(lambda);
energy2=lambda(2)/sum(lambda);
energy3=lambda(3)/sum(lambda);

% The first mode is dominating 99.05% of the energy.
xx = U(:,1)*S(1,1)*V(:,1)';
% We can see xx and X are close.

%%

figure(2)
% we also present all five modes here.
for j = 1:5
    xx = U(:,1:j)*S(1:j,1:j)*V(:,1:j)';
    for k = 1:6
        subplot(6,5,j+(k-1)*5)
        plot([1:numFrames], X(k,:), 'r-',[1:numFrames],xx(k,:),'k-');
        set(gca,'Ylim',[100 500]);
        legend('Original X or Y', ['Mode 1:' num2str(j)]);
    end
end

% We can see from the graph, the first three modes can capture most of the
% features of X.

%%

figure(3)
plot([1:6],U(:,1),'k',[1:6],U(:,2),'k--',[1:6],U(:,3),'k:','Linewidth',[2]);
legend('mode 1','mode 2','mode 3');
