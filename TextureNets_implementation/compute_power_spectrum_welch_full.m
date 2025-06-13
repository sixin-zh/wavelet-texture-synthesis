function [pos,kur]=compute_power_spectrum_welch_full(imgs,wJ)
    addpath ../kymatio_wph3/scatnet-0.2a
    addpath_scatnet;
    
    wJ=double(wJ);
    if length(size(imgs))==3
        K=size(imgs,1);
        N=size(imgs,2);
        assert(size(imgs,3)==N);
    else
        K=1;
        N=size(imgs,1);
        assert(size(imgs,2)==N);
        imgs=reshape(imgs,1,N,N);
    end
    M = floor(2^wJ);
    phiJ=M*M*compute_phiJ_hanning(M);
    % full size as imgs	
    onu=zeros(N,N);
    oh=zeros(N,N);
    Kk=0;
    for k=1:K
        cimg=reshape(imgs(k,:,:),N,N);
        % crop into half overlapping patches, each patch size is MxM
        for cx=[M/2:M/2:N-M/2]
            for cy=[M/2:M/2:N-M/2]
                oimg=phiJ.*crop(cimg,M,[cx,cy]);
                offtabs=abs(fft2(oimg,N,N))/M; % check /M vs /N
                oh=oh+offtabs.^2;
                onu=onu+offtabs.^4;
                Kk=Kk+1;
            end
        end
    end
    oh=oh/Kk;
    onu=onu/Kk;
    
    okurtosis = onu ./ ((oh.^2));
    pos=oh;
    kur=okurtosis;
end
