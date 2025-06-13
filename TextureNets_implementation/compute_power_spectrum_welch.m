function [pos,kur]=compute_power_spectrum_welch(imgs,wJ)
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
%    ds=log2(N)-wJ; M=N/2^ds;
    M = floor(2^wJ);
    phiJ=M*M*compute_phiJ_hanning(M);
%     phiJ=M*M*(compute_phiJ(N,wJ,ds,sigma));
    onu=zeros(M,M);
    oh=zeros(M,M);
    Kk=0;
    for k=1:K
        cimg=reshape(imgs(k,:,:),N,N);
        % crop into half overlapping patches, each patch size is MxM
        for cx=[M/2:M/2:N-M/2]
            for cy=[M/2:M/2:N-M/2]
                oimg=phiJ.*crop(cimg,M,[cx,cy]);
                offtabs=abs(fft2(oimg))/M;
                oh=oh+offtabs.^2;
                onu=onu+offtabs.^4;
                Kk=Kk+1;
            end
        end
    end
    oh=oh/Kk;
    onu=onu/Kk;
    
%     osig=sqrt(onu-oh.^2);
%     osnr=oh./osig;
%     okurtosis=1./(osnr.^2)+1;
    okurtosis = onu ./ ((oh.^2));
%     [oS,oV,oK]=mySpectre2D(oh);
    pos=oh;
    kur=okurtosis;
%     rS=oS;
%     rV=oV;
%     rK=oK;
end