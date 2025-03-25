function plotIm(W)

% function plotIm(W)
%
% Plots square images reshaping
%
% INPUTS
% W = generative weights [D,K]

%LoadFigureSettings;  
    
[D,K] = size(W);  
%[D,DD] = size(G);  

NumPlots = ceil(sqrt(K));
NumPix = sqrt(D);

top = 0.05;
bottom = 0.05;
left = 0.05;
right = 0.05;
vspace = 0.01;
hspace = 0.01;

width = (1-left-right-hspace*(NumPlots-1))/NumPlots;
height = (1-top-bottom-vspace*(NumPlots-1))/NumPlots;

across = [width+hspace,0,0,0]';
down = -[0,height+vspace,0,0]';

pos = zeros(4,NumPlots,NumPlots);

for d1=1:NumPlots
  for d2=1:NumPlots
    pos(:,d1,d2) = [left, 1-top-height,width,height]' ...
                   + (d1-1)*across+(d2-1)*down;
  end
end

pos = reshape(pos,[4,NumPlots*NumPlots]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fix size of figure so that the patches are square

ScrSz = get( 0, 'ScreenSize' );

hFrac = 0.8;
hFig = ScrSz(4)*hFrac;
wFig = height/width*hFig;

posFig = [ScrSz(3)/2-wFig/2,ScrSz(4)/2-hFig/2,wFig,hFig];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WAll = W';
%clim = [min(WAll(:)),max(WAll(:))];
clim = max(abs(WAll(:)))*[-1,1];

figure1 = figure('Name','Weights','NumberTitle','off','position',posFig);

for k=1:K
  %subplot(NumPlots,NumPlots,k);

  axk = axes('position',pos(:,k));
  hold on;
  
  Wcur = W(:,k)';
  
  imagesc(reshape(Wcur',[NumPix,NumPix]),max(abs(Wcur))*[-1,1]+[-1e-5,1e-5])
%  imagesc(reshape(Wcur',[NumPix,NumPix]),clim)
  set(gca,'ylim',[1,NumPix],'xlim',[1,NumPix])
  set(gca,'yticklabel','','xticklabel','','visible','off')
%  box off
  colormap gray
  

end  
  
