close all
clear all

addpath ../altmany-export_fig-3.40.0.0 

% compute radial power spectrum from samples of ori and model
name = 'tur2a';
% name = 'fbmB7';

% model = 'linConv2d_modelA';
% model = 'linIdwt2d_modelA';
% model = 'linIdwt2d_modelA_djl';
% model = 'linIdwt2d_modelA_adam';
% model = 'linIdwt2d_modelA_altgda';

% model = 'modelA';
% model = 'modelC';
% model = 'modelC_adam';
model = 'vgg_gray';

% model = 'textureNets_g2d_vgg_gray';
% model = 'textureNets_g2d_modelC';
% model = 'textureNets_g2d_modelA';
% model = 'textureNets_g2d_lin_modelA';
% model = 'g2d_modelA_altgda';
% model = 'g2d_modelC_altgda';

fprintf('plot name=%s,model=%s\n',name,model);

use_welch = 1;

switch name
    case 'tur2a'
        load('../turbulence/ns_randn4_test_N256.mat') % from ori, get imgs
    case 'fbmB7'
        load('../turbulence/demo_fbmB7_N256.mat') % from ori, get imgs
end
switch model
    case 'modelA'
        Krecs = 2;
        tkt = 'bump_lbfgs2_gpu_N256J5L8dn2_factr10maxite500maxcor20_initnormal_ks0ns1'
        model_dir = sprintf("./results_acha/%s/%s/",name,tkt);
        model_id = "modelA";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'modelC'
        %Krecs = 2;
        %model_dir = sprintf("./results_acha/%s/bump_lbfgs2_gpu_N256J5L8dj1dl4dk0dn2_maxkshift1_factr10maxite500maxcor20_initnormal_ks0ns1/",name);
        
        Krecs = 1;        
        % tkt = 'N=256&ks=0&J=5&L=8&dj=1&dl=4&dk=0&dn=2&maxk=1&factr=10.0&maxite=500&maxcor=20&init=normal&ns=1&gpu=True';
        %tkt = 'N=256&ks=0&J=5&L=8&dj=1&dl=4&dk=0&dn=2&maxk=1&factr=10.0&maxite=500&maxcor=20&init=normalstdbarx&ns=1&gpu=True';
        
        tkt = 'N=256&ks=0&J=5&L=8&dj=1&dl=4&dk=0&dn=2&maxk=1&factr=10.0&maxite=500&lr=0.1&init=normalstdbarx&ns=1&gpu=True';
        
        model_dir = sprintf("./results_acha/%s/%s/",name,tkt);        
        model_id = "modelC";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'modelC_adam'
        Krecs = 1;              
        tkt = 'N=256&ks=0&J=5&L=8&dj=1&dl=4&dk=0&dn=2&maxk=1&factr=10.0&maxite=500&lr=0.1&init=normal&ns=1&gpu=True';
        model_dir = sprintf("./results_acha/%s/%s/",name,tkt);        
        model_id = "modelC";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'vgg_gray'
        Krecs = 1;
        % tkt = 'N=256&vggl=3&Ns=256&its=2000&factr=10.0&runid=1&gpu=True';
        tkt = 'N=256&vggl=3&Ns=1024&its=30&factr=10.0&runid=1&gpu=True';
        
        model_dir = sprintf("./results_vgg_model_gray/%s/%s/",name,tkt);        
        model_id = "tur2a_im0_vgg_nohist";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'linConv2d_modelA'
        Krecs = 16;
        if strcmp(name,'fbmB7')==1
            model_dir = "./ckpt/fbmB7_LinConv2D/J=5&L=8&fs=101&dn=2&lr=0.01&its=500&bs=16&factr=1.0&runid=1&init=normalstdbarx&spite=10&gpu=True/";
            model_id = "training_500";
        elseif strcmp(name,'tur2a')==1
            model_dir = "./ckpt/tur2a_LinConv2D/J=5&L=8&fs=81&dn=2&lr=0.01&its=500&bs=16&factr=1.0&runid=1&init=normalstdbarx&spite=10&gpu=True/";
            model_id = "trained_samples";
        end
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'linIdwt2d_modelA'
        Krecs = 16;
        % tkt = 'J=5&L=8&dn=2&lr=0.01&its=500&fs=64&bs=16&factr=100.0&runid=2&init=normal&gpu=True&loaddir=0'; #% fs 64
	% fs 128
        % model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA/J=5&L=8&dn=2&lr=0.01&its=500&fs=128&bs=16&factr=100.0&runid=1&init=normal&gpu=True&loaddir=0/",name);
	% model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA/J=5&L=8&dn=4&lr=0.01&its=500&fs=64&bs=16&factr=100.0&runid=1&init=normal&gpu=True&loaddir=0/",name);
	% tkt = 'J=5&L=8&dn=4&lr=0.01&its=500&fs=64&bs=16&factr=100.0&runid=1&init=normal&gpu=True&loaddir=0' % with fs 64, dn 4
	% tkt = 'J=5&L=8&dn=2&lr=0.01&its=500&fs=256&bs=16&factr=100.0&runid=1&init=normal&gpu=True&loaddir=0' % with fs 256, dn 2
%        tkt = 'J=5&L=8&dn=2&its=500&fs=64&bs=16&factr=10.0&runid=1&init=normal&gpu=True&loaddir=0&wave=db7'; % with db7
%        tkt = 'J=5&L=8&dn=2&its=500&fs=64&bs=16&factr=10.0&runid=1&init=normal&gpu=True&loaddir=0&wave=db9'; % with db9
%        tkt = 'J=5&L=8&dn=2&its=500&fs=64&bs=16&factr=10.0&runid=1&init=normal&gpu=True&loaddir=0&wave=db11'; % with db11
        tkt = 'J=5&L=8&dn=2&its=500&fs=64&bs=16&factr=10.0&runid=1&init=normal&gpu=True&loaddir=0&wave=db13'; % with db13
        model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA/%s/",name,tkt);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model	  
        
    case 'linIdwt2d_modelA_altgda'
        % Krecs = 1;	  
        % tkt = 'J=5&L=8&dn=2&its=5000&lrD=0.1&lrG=0.1&tau=5&bs=1&fs=64&factr=10.0&spite=10&runid=1&wave=db7&init=normal&gpu=True&loaddir=0';
	tkt = 'J=5&L=8&dn=2&its=5000&lrD=0.1&lrG=0.1&tau=5&bs=16&fs=64&factr=10.0&spite=10&runid=1&wave=db7&init=normal&gpu=True&loaddir=0';
	Krecs = 16;
        model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA_altgda/%s",name,tkt)
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model   
        
    case 'linIdwt2d_modelA_djl'
        Krecs = 10;	    
	tkt = 'J=5&L=8&fs=64&dn=2&dj=1&dl=1&its=500&bs=10&factr=100.0&runid=1&init=normal&gpu=True&loaddir=0'
        % with supp moments
        model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA_djl/%s",name,tkt)
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'linIdwt2d_modelA_adam'
        Krecs = 16;
        tkt = 'J=5&L=8&dn=2&its=1000&lr=0.01&bs=16&fs=64&factr=10.0&spite=10&runid=1&init=normal&gpu=True&loaddir=0';
        model_dir = sprintf("./ckpt/%s_linIdwt2d_modelA_adam/%s",name,tkt);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
        
    case 'textureNets_g2d_modelC'
        %Krecs = 16;
        %model_dir = sprintf("./ckpt/%s_g2d_modelC/J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&lr=0.0001&its=500&bs=16&factr=1.0&runid=1&init=normal&spite=10&gpu=True&loaddir=1/",name);
        
        Krecs = 1;
        tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=16&lr=0.001&its=10000&bs=1&resample=1&factr=10.0&runid=1&init=normalstdbarx&spite=10&gpu=True&loaddir=0';
        model_dir = sprintf("./ckpt/%s_g2d_modelC/%s",name,tkt);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model        
        
    case 'textureNets_g2d_modelA'
        Krecs = 16;
        model_dir = sprintf("./ckpt/%s_g2d/J=5&L=8&fs=101&dn=2&lr=0.01&its=500&bs=16&factr=1.0&runid=1&init=normalstdbarx&spite=10&gpu=True/",name);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model   
        
    case 'textureNets_g2d_lin_modelA'
        Krecs = 16;
%        model_dir = sprintf("./ckpt/%s_g2d_lin_modelA/J=5&L=8&dn=2&lr=0.01&its=500&ch=1&bs=16&factr=10.0&runid=1&init=normal&spite=10&gpu=True&loaddir=0",name);
%         model_dir = sprintf("./ckpt/%s_g2d_lin_modelA/J=5&L=8&dn=2&lr=0.005&its=500&ch=1&bs=16&factr=10.0&runid=1&init=normal&spite=10&gpu=True&loaddir=0",name);        
        model_dir = sprintf("./ckpt/%s_g2d_lin_modelA/J=5&L=8&dn=2&lr=0.005&its=2000&ch=1&bs=16&factr=10.0&runid=1&init=normal&spite=10&gpu=True&loaddir=0",name);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model   
        
    case 'textureNets_modelA'
        Krecs = 2;
        assert(false)
        
    case 'g2d_modelA_altgda'
        Krecs = 1;
        tkt = 'J=5&L=8&dn=2&its=5000&lrD=0.1&lrG=0.01&tau=5&bs=1&ch=8&factr=10.0&spite=10&runid=1&wave=db3&init=normal&gpu=True&loaddir=0';
        model_dir = sprintf("./ckpt/%s_g2d_modelA_altgda/%s",name,tkt);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model        
        
    case 'g2d_modelC_altgda'
        Krecs = 1; 
        % tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&its=5000&lrD=0.1&lrG=0.01&tau=5&bs=1&ch=8&factr=10.0&spite=10&runid=1&wave=db3&init=normal&gpu=True&loaddir=0';
        % model_dir = sprintf("./ckpt/%s_g2d_modelA_altgda/%s",name,tkt);
%        tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&rand=1&its=5000&lrD=0.1&lrG=0.005&tau=5&bs=1&factr=10.0&spite=10&runid=1&init=normal&gpu=True&loaddir=0';
        % tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=8&rand=1&its=50000&lrD=0.1&lrG=0.01&tau=50&bs=1&factr=10.0&spite=100&runid=1&init=normal&gpu=True&loaddir=0';
        % tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=16&rand=0&its=5000&lrD=0.01&lrG=0.0001&tau=20&bs=1&factr=10.0&spite=10&runid=1&init=normalstdbarx&gpu=True&loaddir=0';
        % tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=16&rand=0&its=5000&lrD=0.01&lrG=1e-05&tau=20&bs=1&factr=10.0&spite=10&runid=2&init=normalstdbarx&gpu=True&loaddir=0';
        tkt = 'J=5&L=8&dn=2&dj=1&dk=0&dl=4&ch=16&rand=0&its=5000&lrD=0.001&lrG=1e-05&tau=20&bs=1&factr=10.0&spite=10&runid=1&init=normalstdbarx&gpu=True&loaddir=0';
        
        model_dir = sprintf("./ckpt/%s_g2d_modelC_altgda/%s",name,tkt);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model
    
    case 'textureNets_g2d_vgg_gray'
        Krecs = 16;
        model_dir = sprintf("./ckpt/%s_g2d_gray/N=256&ks=0&lr=0.01&its=10000&bs=10&factr=10&runid=1&spite=100&gpu=True",name);
        model_id = "trained_samples";
        ofile = sprintf('%s/%s.mat',model_dir,model_id);  % from model           
end

% psd of original, est from K=2 sample
N = size(imgs,1);
K = size(imgs,3);
wJ = log2(N)-1;

if use_welch == 1
    imgs_ = permute(imgs,[3,1,2]);
    [hatK0,kur]=compute_power_spectrum_welch(imgs_,wJ);
    M = size(hatK0,1);
else
    spImgs = zeros(N,N,K);
    for k=1:K
        spImgs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);
    end
    hatK0 = mean(spImgs,3);
%     spK0 = mySpectre2D(hatK0);
    M = N;
end
spK0 = mySpectre2D(hatK0);
    
%% psd of synthesis, est from Krecs samples
load(ofile); % get imgs
if use_welch == 1
    imgs_ = permute(imgs,[3,1,2]);
    [hatKr,kur]=compute_power_spectrum_welch(imgs_,wJ);
else
    spRecs = zeros(N,N,Krecs);
    for k=1:Krecs
        if Krecs > 1
            spRecs(:,:,k)=(abs(fft2(imgs(:,:,k))).^2)/(N^2);      
        else
            spRecs(:,:,k)=(abs(fft2(imgs(:,:))).^2)/(N^2);
        end
    end
    hatKr = mean(spRecs,3);
end
spXrec = mySpectre2D(hatKr);

%%  plot radius spectrum
figure(1);
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

%% 2d spectrum
figure(2);
loghX=log10(hatK0);
loghXrec = log10(hatKr);
inrag=[min(loghXrec(:)),max(loghXrec(:))];
%         inrag=[min(loghX(:)),max(loghX(:))];
subplot(121)
imagesc(fftshift(loghX),inrag); colorbar; axis square
title('Empirical','FontSize',20)
subplot(122)
imagesc(fftshift(loghXrec),inrag); colorbar; axis square
title('Model','FontSize',20)
% : log10 P(\omega)
axis tight
export_fig(sprintf('curves/ps_%s_%s.pdf',model,name),'-pdf','-transparent')  
