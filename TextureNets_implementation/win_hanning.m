% H=4;
% q=-H/2:H/2-1;
% w=cos(pi*q/H)*sqrt(2/H);
% sum(w.^2)
H=8; % data win size
L=8; % angles [0,pi)
% assert(mod(L,H)==0)
% check if v forms a frame, sum v^2 = 1
% check if w = v * tilde(v) is regular

j=0:H;
v=0.5*(1-cos(2*pi*(j)/H));
% v=cos(pi*(j-H/2)/H)/sqrt(2/H);
P=mean(v.*v)

v=v./sqrt(2*sum(v.*v));
% v=v./sum(v);

% w=zeros(size(v));
% for k=0:H-1
%     for n=k+1:H
%         w(k+1)=w(k+1)+v(n)*v(n-k);
%     end
% end
% w=w./H

% whanning=1+cos(pi*j/H);
% whanning=whanning./whanning(1)*w(1);
% 
% plot(w)
% hold on
% plot(whanning)
% hold off

%%
L2=2*L;
Delta=floor(H/2);
assert(mod(L2,Delta)==0)
W=zeros(H*L2/Delta,L2);
k=0;
for l=0:H-1
    for n=0:L2/Delta-1
        for q=0:L2-1
            qn=mod((q-n*Delta),L2);
            theta=l*qn*2*pi/H;
            if qn>=0 && qn<=H-1
                wn=v(qn+1);
            else
                wn=0;
            end
            W(k+1,q+1)=wn*(cos(theta)-1i*sin(theta));
        end
        k=k+1;
    end
end

GramW=W*W';
eig(GramW)

% figure
subplot(121)
hold on
plot(abs(W(1,:)))
subplot(122)
hold on

plot((abs(fft(W(1,:)))))

%% random test
x=randn(L2,1);x=x/norm(x);
y=W*x;
norm(y)