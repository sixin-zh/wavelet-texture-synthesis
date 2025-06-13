% compare peridogram with power spectrum of a stationary process
% normalize with diag terms
close all
clear all

addpath ../altmany-export_fig-3.40.0.0

name = 'fbmB7';
model = 'linIdwt2d_modelA';
runid = 1;

% model = 'linIdwt2d_modelA_altgda';
% runid = 1009;

Kfold = 1;

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

spImgs = zeros(N,N,K);
for k=1:K
    spImgs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);
end
hatK0 = mean(spImgs,3);

hatD0 = sqrt(hatK0);

% estimate power spectrum from textureNet samples, trained on model A moments

modelA_max_error = zeros(Kfold,1);
modelA_mean_error = zeros(Kfold,1);
for fold = 1:Kfold    
    kid = fold;
    [ofile,Krecs] = get_ofile(name,model,runid);
    load(ofile); % get imgs
    spImgs = zeros(N,N,Krecs);
    for k=1:Krecs
        spImgs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);
    end
    hatKr = mean(spImgs,3);

    rel_err = abs((hatKr ./ hatK0 - 1));
    modelA_mean_error(fold) = mean(mean(rel_err)); % max(abs((hatKr ./ hatK0 - 1))));

    [maxW,idxH] = max(rel_err);
    % disp(size(maxW)) % ,idxW)
    [max_err,idxW] = max(maxW);
    modelA_max_error(fold) = max_err; % max(abs((hatKr ./ hatK0 - 1))));
    fprintf('max: attained at freq (%g,%g) for fold %d\n',2*pi*idxH(idxW)/N,2*pi*idxW/N,fold); 
    % disp(hatKr(idxH(idxW),idxW) / hatK0(idxH(idxW),idxW))
end

fprintf('max: mean/std is %g (%g)\n',mean(modelA_max_error),std(modelA_max_error));
fprintf('mean: mean/std is %g (%g)\n',mean(modelA_mean_error),std(modelA_mean_error));



%%  plot radius spectrum
spK0 = mySpectre2D(hatK0);
spXrec = mySpectre2D(hatKr);
M = N;

figure;
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
export_fig(sprintf('curves/radialps_%s_%s_%d.pdf',model,name,runid),'-pdf','-transparent')  


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
export_fig(sprintf('curves/ps_%s_%s_%d.pdf',model,name,runid),'-pdf','-transparent')


function [ofile,Krecs] = get_ofile(name,model,runid) % J,L,dn,bs,ks,runid,ebs,T
    switch model
        case 'linIdwt2d_modelA'
		if runid == 1
			tkt = 'J=5&L=4&dn=1&wave=db7&gJ=4&fs=32&its=100&bs=1024&ebs=100&ks=0&factr=10.0&init=normalstdonly&runid=1&loaddir=0'
	                model_id = "eval_samples";
			Krecs =100;
		else
			tkt = '';
		end
%            if J == 5
%                assert(ks==0)
%                tkt = 'J=5&L=8&dn=2&its=500&fs=64&bs=16&factr=10.0&runid=2&init=normal&gpu=True&loaddir=0&wave=db7';
%                Krecs = 16;
%                model_id = "trained_samples";
%            elseif J == 4
%                tkt = sprintf('J=4&L=4&dn=%d&its=500&fs=32&bs=%d&factr=10.0&runid=%d&init=normal&gpu=True&ks=%d&loaddir=0&wave=db7',dn,bs,runid,ks);
%                Krecs = bs;
%                model_id = "test_samples";
%            end
            model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA/%s",name,tkt);
            ofile = sprintf('%s/%s.mat',model_dir,model_id)  % from model
        case 'linIdwt2d_modelA_altgda'
%             tkt = sprintf('J=%d&L=%d&dn=%d&init=normalstdonly&fs=32&gJ=4&wave=db7&ks=%d&bs=%d&ebs=%d&its=%d&lrD=0.1&lrG=0.1&tau=5&factr=10.0&runid=%d&loaddir=0',...
%                           J,L,dn,ks,bs,ebs,T,runid);
            if runid == 1002
                tkt = 'J=5&L=4&dn=1&init=normal&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.1&lrG=0.1&tau=5&factr=10.0&runid=1002&loaddir=0';
                Krecs = 100;
            elseif runid == 1003
                tkt = 'J=5&L=4&dn=1&init=normal&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.01&lrG=0.01&tau=5&factr=10.0&runid=1003&loaddir=0';
                Krecs = 100;
            elseif runid == 1004
                tkt = 'J=5&L=4&dn=1&init=normal&fs=64&gJ=5&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.01&lrG=0.01&tau=5&factr=10.0&runid=1004&loaddir=0';
                Krecs = 100;
            elseif runid == 1005
                tkt = 'J=5&L=4&dn=1&init=normal&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.01&lrG=0.001&tau=5&factr=10.0&runid=1005&loaddir=0';
                Krecs = 100;
            elseif runid == 1006
                tkt = 'J=5&L=4&dn=1&init=normalstdonly&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.01&lrG=0.001&tau=5&factr=10.0&runid=1006&loaddir=0';
                Krecs = 100;
            elseif runid == 1007
                tkt = 'J=5&L=4&dn=1&init=normalstdonly&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=5000&lrD=0.01&lrG=0.005&tau=5&factr=10.0&runid=1007&loaddir=0';
                Krecs = 100;
            elseif runid == 1009
                tkt = 'J=5&L=4&dn=1&init=normalstdonly&fs=32&gJ=4&wave=db7&ks=0&bs=16&ebs=100&its=50000&lrD=0.01&lrG=0.001&tau=5&factr=10.0&runid=1009&loaddir=0';
                Krecs = 100;
            else
                J = 5;
                L = 4;
                dn = 1;
                bs = 16;
                ebs = 100;
                runid = 1001;
                T = 5000;
                tkt = sprintf('J=%d&L=%d&dn=%d&init=normalstdonly&fs=32&gJ=4&wave=db7&ks=%d&bs=%d&ebs=%d&its=%d&lrD=0.1&lrG=0.01&tau=5&factr=10.0&runid=%d&loaddir=0',...
                   J,L,dn,ks,bs,ebs,T,runid);
            end
            model_id = "eval_samples";
            if runid < 1000
                model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA_altgda/%s",name,tkt);
            else
                model_dir = sprintf("./ckpt_calmip/%s_linIdwt2d_modelA_altgda/%s",name,tkt);
            end
            ofile = sprintf('%s/%s.mat',model_dir,model_id)  % from model   
    end
end

