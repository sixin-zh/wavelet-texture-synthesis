% rotational invariant 2d welch win
function phiJ=compute_phiJ_hanning(M)
        
    addpath ./window2
    phiJ=window2(M,M,@hann);
    
%     j=0:M-1;
%     v=0.5*(1-cos(2*pi*(j)/M));
%     
%     A=zeros(M,M);
%     siz=size(A);
%     [y,x] = meshgrid(1:siz(2),1:siz(1));
%     x=x-(siz(1)/2+1);
%     y=y-(siz(2)/2+1);
%     modx=sqrt(x.^2 + y.^2);
%     
%     K=0:(min(siz(1),siz(2))/2)-1;
%     
%     for i=K
%         mask = ((modx>=i)&(modx<(i+1)));
% %         [I,J]=find(mask)
%         A(mask)=v(M/2+i+1);
%     end
%     
%     phiJ=A;
    phiJ=phiJ./norm(phiJ(:));
    phiJ=phiJ/M;
    
%    	subplot(2,1,1)
%     imagesc(phiJ)
%     subplot(2,1,2)
%     imagesc(fftshift(log(abs(fft2((phiJ))))))
end
