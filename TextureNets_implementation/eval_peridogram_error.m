% compare peridogram with power spectrum of a stationary process
% normalize with diag terms
close all
clear all

% addpath ../pwelch2
addpath ../altmany-export_fig-3.40.0.0

name = 'fbmB7';
model = 'periodigram';

use_welch = 1;

% estimate power spectrum from test data
switch name
    case 'tur2a'
        load('../turbulence/ns_randn4_test_N256.mat') % from ori, get imgs
    case 'fbmB7'
        load('../turbulence/fbmB7_test_N256.mat') % from ori, get imgs
end

% compute hatK0 from test samples
N = size(imgs,1);
K = size(imgs,3);

% set window size
if use_welch ==1
    wJ = log2(N)-3;
    %wJ = log2(N)-2;
    model = 'welch';
else
    model = 'periodigram';
end

%if use_welch == 1
%    imgs_ = permute(imgs,[3,1,2]);
%    [hatK0,kur]=compute_power_spectrum_welch_full(imgs_,wJ);
%    M = size(hatK0,1);
%else
spImgs = zeros(N,N,K);
for k=1:K
    spImgs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);
end
hatK0 = mean(spImgs,3);
M = N;
%end

hatD0 = sqrt(hatK0);

% estimate power spectrum from training data
% use only 1 training sample (kid is idx of training image)
% from ori, get imgs
load('../turbulence/fbmB7_train_N256.mat')
Kfold = 10;
periodogram_max_error = zeros(Kfold,1);
periodogram_mean_error = zeros(Kfold,1);

for fold = 1:Kfold
    kid = fold;
    Krecs = 1;
    if use_welch == 1
        imgs_ = zeros(1,N,N); % permute(imgs,[3,1,2]);
	imgs_(1,:,:) = imgs(:,:,kid);
        [hatKr,kur]=compute_power_spectrum_welch_full(imgs_,wJ);
    else
        spRecs = zeros(N,N,Krecs);
        spRecs(:,:,1)=(abs(fft2(imgs(:,:,kid))).^2)/(N^2);      
        hatKr = mean(spRecs,3);
    end

    rel_err = abs((hatKr ./ hatK0 - 1));
    periodogram_mean_error(fold) = mean(mean(rel_err)); % max(abs((hatKr ./ hatK0 - 1))));

    [maxW,idxH] = max(rel_err);
    % disp(size(maxW)) % ,idxW)
    [max_err,idxW] = max(maxW);
    periodogram_max_error(fold) = max_err; % max(abs((hatKr ./ hatK0 - 1))));
    fprintf('max: attained at freq (%g,%g) for fold %d\n',2*pi*idxH(idxW)/N,2*pi*idxW/N,fold);

    % periodogram_error(fold) = max(max(abs((hatKr ./ hatK0 - 1))));
end
%fprintf('mean/std is %g (%g)\n',mean(periodogram_error),std(periodogram_error));

fprintf('max: mean/std is %g (%g)\n',mean(periodogram_max_error),std(periodogram_max_error));
fprintf('mean: mean/std is %g (%g)\n',mean(periodogram_mean_error),std(periodogram_mean_error));


%% 2d spectrum
figure();
loghX=log10(hatK0);
loghXrec = log10(hatKr);
% inrag=[min(loghXrec(:)),max(loghXrec(:))];
inrag=[min(loghX(:)),max(loghX(:))];
subplot(121)
imagesc(fftshift(loghX),inrag); colorbar; axis square
title('Empirical','FontSize',20)
subplot(122)
imagesc(fftshift(loghXrec),inrag); colorbar; axis square
title('Model','FontSize',20)
% : log10 P(\omega)
axis tight
export_fig(sprintf('curves/ps_%s_%s.pdf',model,name),'-pdf','-transparent')

%%  plot radius spectrum
spK0 = mySpectre2D(hatK0);
spXrec = mySpectre2D(hatKr);

figure();
xk = 0:pi/(M/2):pi;
xk = xk(2:end);
plot(xk,log10(spK0),'b-')
hold on
plot(xk,log10(spXrec),'b--')

set(gca,'fontsize',24)
xticks([0.1,1,pi])
xticklabels({'10^{-1}','10^{0}','\pi'})
hold off
set(gca,'xscale','log')

xlabel('$$|\omega|$$','interpreter','latex')
axis tight
export_fig(sprintf('curves/radialps_%s_%s.pdf',model,name),'-pdf','-transparent')  
