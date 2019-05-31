function [IMF,SDlog,havelog] = Decomp_MIF_2D_v12(f,options,alpha)
%
% Generate the decomposition of a 2D signal f :
%
%  f = IMF(:,:,1) + IMF(:,:,2) + ... + IMF(:, :,size(IMF, 3))
%
% where the last component is the trend and other components are IMFs
% The mask length is computed based on alpha and the mask length values 
% from the horizontal middle section of the signal
%
%                    Input
%
%   f          2D signal to be decomposed
%
%   options    Structure, generated using function decompSettings_MIF_2D_v10, containing
%              all the parameters needed in the various algorithms
%
%   alpha      Allows to tune the mask length between the minimum and
%              maximum distance between two consecutive extreme points in
%              the middle section of the signal. If alpha = 1 the mask length
%              is set equal to the maximum distance between two subsequent
%              extrema. If alpha = 0 then it equals the minimum distance.
%              Default value is ('ave') which equals the average mask
%              length of the middle section of the signal. 
%              Allowed values [0,1] and ('ave').
%
%   See also DECOMPSETTINGS_MIF_2D_V10, MAXMINS, GETMASK_2D_V1.
%
%  Ref: A. Cicone, H. Zhou. 'Multidimensional Iterative Filtering method 
%      for the decomposition of high-dimensional non-stationary signals'.
%      Preprint ArXiv http://arxiv.org/abs/1507.07173
% 


%% deal with the input

if nargin == 0,  help Decomp_MIF_2D_v10; IMF=[];SDlog=[];havelog=[];return; end
if nargin == 1, alpha='ave'; options = decompSettings_MIF_2D_v01; end
if nargin == 2, options = decompSettings_MIF_2D_v01; end

%FigCol = 'ckmygr'; % Plot Colors


N = size(f);

IMF =[];

if options.saveplots>0
    nameFile=input('Please enter the name of the file as a string using '' and '' <<  ');
else
    nameFile=['Test_' datestr(clock,'YYYY_mm_DD_HH_MM_SS')];
end

%% Main code
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                 Iterative Filtering                %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    load('prefixed_double_filter','MM');
    
    nIMFs=0;
    while nIMFs < options.MIF.NIMFs && toc<options.maxTime
        nIMFs=nIMFs+1;
        if options.verbose>0
            fprintf('\n IMF # %1.0d\n',nIMFs)
        end
        SD=1;
        SDlog=[];
        havelog=[];
        h=f;
        
        %%%%%%%%%%%%%%% Identify extreme points of the middle section %%%%%%%%%%%%%%
        maxmins_f=Maxmins(f(round(end/2),:),'p');
%         figure
%         plot(f(round(end/2),:))
        diffMaxmins_f=diff(maxmins_f);
        k = length(maxmins_f);
        
        if isa(alpha,'char')
            if strcmp(alpha,'ave') % Using an average mask length
                m = round(2*(N(2)/k)*options.MIF.Xi);
            else
                disp(' Value of alpha not recognized')
                return
            end
        else % using a prefixed value alpha
            m = 2*round(options.MIF.Xi*(max(diffMaxmins_f)*alpha+min(diffMaxmins_f)*(1-alpha)));
        end
        
        
        
        
        inStepN=0;
        if options.verbose>0
            fprintf('\n  step #            SD             Mask length \n\n')
        end
        
        A = get_mask_2D_v1(MM,m);
        if options.plots>0
            ImgIMF=figure;
            ImgR=figure;
		end
		
		if options.MIF.fft
			% Precompute the FFT of the signal and of the filter, so we can
			% apply it with almost no computations
			[fftm, fftn] = size(h);
			[fftp, fftq] = size(A);
			
			% Note that if the filter has a support larger than our image
			% we might need to extend the image a little bit. The repmat
			% command takes care of that.
			if fftp > fftm || fftq > fftn
				fftH = repmat(h, ceil(max(fftp/fftm, fftn/fftq)));
				[fftm, fftn] = size(fftH);
			else
				fftH = h;
			end
			
			% Pad A the right way -- this is required to make sure the
			% center of gravity of the filter is in position (1,1) before
			% taking the FFT
			l1 = floor(fftp/2); m1 = floor(fftq/2);
			l2 = fftp - l1; m2 = fftq - m1;
			fftA = zeros(fftm, fftn); 
			fftA(1:l2,1:m2) = A(l1+1:end,m1+1:end);
			fftA(end-l1+1:end,end-m1+1:end) = A(1:l1,1:m1);
			fftA(end-l1+1:end,1:m2) = A(1:l1,m1+1:end);
			fftA(1:l2,end-m1+1:end) = A(l1+1:end,1:m1);
			
			% Precomputing FFTs for later use
			fftH = fft2(fftH);
			fftA = fft2(fftA);
			
			% r1 and r2 are updated throughout the iterations so that r1 /
			% r2 is equal to the relative change in the solution at step j.
			% To accomplish these, some values are accumulated in
			% filter_factors (which indeed describes the action of the
			% accumulated filter on all the frequencies), making use of the
			% variable incr_filter. 
			r2 = abs(fftH).^2;
			r1 = r2 .* abs(fftA).^2;
			incr_filter = (1 - abs(fftA)).^2;
			filter_factors = ones(size(fftA));
		end        
        
        while SD>options.MIF.delta && inStepN < options.MIF.MaxInner
            inStepN=inStepN+1;
            
			% FIXME: Similar to the 1D case, this might be made more
			% efficient skipping the iterations completely, and just
			% computing the necessary index j. 
			if options.MIF.fft
				% Construct the residual for checking the stopping
				% criterion
				h_avenrm = sqrt(sum(sum( r2 .* filter_factors )));
				SD = sum(sum( r1 .* filter_factors )) ./ ...
					 h_avenrm^2;
				 
				% We never explicitly compute h_ave, and we construct the
				% final solution h only at convergence
				if SD < options.MIF.delta || inStepN > options.MIF.MaxInner
					h = real(ifft2( fftH .* (1 - fftA).^(inStepN) ));
				else
					% Update the filter factors for the next round
					filter_factors = incr_filter .* filter_factors;
				end
			else
				h_ave = conv2(h, A, 'same');
				
				h_avenrm = norm(h_ave, 'fro');
				SD=h_avenrm^2 / norm(h, 'fro')^2;
			end
            
            %%%%%%%%%%%%%%%% Updating stopping criterium %%%%%%%%%%%%%%%%%
            
            SDlog=[SDlog SD];
            havelog=[havelog h_avenrm];
            if options.verbose>0
                fprintf('    %2.0d      %1.14f          %2.0d\n',inStepN,SD,m)
            end
            
            %%%%%%%%%%%%%%%%%% generating f_n %%%%%%%%%%%%%%%%%%
            
			% If using the FFT, then we do not need to subtract the
			% average, which is not computed explicitly. The function h
			% will be automatically constructed at convergence by a direct
			% approach applying the filter with FFT (1 - fftA).^j. 
			if ~options.MIF.fft
				h=h-h_ave;
			end
			
            if options.plots>0
                figure(ImgIMF)
                surfH=surf(h);
                set(surfH, 'edgecolor','none')
                title('IMF')
                pause(0.5)
                %%
                figure(ImgR)
                surfF=surf(f-h);
                set(surfF, 'edgecolor','none')
                title('Remainder')
                pause(0.5)
            end
            if options.saveIntermediate == 1
                save([ nameFile '_inter_decomp_MIF_2D_v10.mat'])
            end            
        end
        
        if inStepN >= options.MIF.MaxInner
            disp('Max # of inner steps reached')
            return
        end 
        
        close all
        
        IMF(:,:,nIMFs) = h;
        f=f-h;
        
%         figure
%         plot(f(round(end/2),:))
%         pause
        
    end
    
    IMF(:,:,end+1) = f;


if options.saveEnd == 1
    save([ nameFile '_decomp_MIF_2D_v10.mat'])
end

end


%% Auxiliar functions


function A=get_mask_2D_v1(w,k)
% get the mask with length 2*k+1 x 2*k+1
% k could be an integer or not an integer
% w is the area under the curve for each bar

n=length(w);
m=(n-1)/2;

if k<=m % The prefixed filter contains enough points
    
    if mod(k,1)==0     % if the mask_length is an integer
        
        A=zeros(k+1,k+1);
        
        N=(m+1)/(k+1);
        w=[w zeros(1,m+1)];
        %         for i=1:k+1
        %             s=(i-1)*N+1;
        %             t=i*N;
        %
        %             %s1=s-floor(s);
        %             s2=ceil(s)-s;
        %
        %             t1=t-floor(t);
        %             %t2=ceil(t)-t;
        %
        %             if floor(t)<1
        %                 disp('Ops')
        %             end
        %             A(1,i)=sum(w(m+ceil(s):m+floor(t)))+s2*w(m+ceil(s))+t1*w(m+floor(t));
        %         end
        for i=1:k+1
            for j=i:k+1
                if i==1 && j==1
                    
                    t=N/2;
                    
                    t1=t-floor(t);
                    
                    if floor(t)<1
                        disp('Ops')
                    end
                    A(i,j)=2*(sum(w(m+2:m+floor(t)))+t1*w(m+floor(t))+w(m+1));
                else
                    d=sqrt((i-1)^2+(j-1)^2);
                    s=(d-1)*N+1;
                    t=d*N;
                    
                    %s1=s-floor(s);
                    s2=ceil(s)-s;
                    
                    t1=t-floor(t);
                    %t2=ceil(t)-t;
                    
                    if floor(t)<1
                        disp('Ops')
                    end
                    A(i,j)=sum(w(m+ceil(s):m+floor(t)))+s2*w(m+ceil(s))+t1*w(m+floor(t));
                end
            end
        end
        A=A'+A-diag(diag(A)); % we complete the fourth quadrant
        A=[rot90(A(2:end,2:end),2) rot90(A(:,2:end),1); rot90(A(2:end,:),3) A]; % Final filter size 2*k+1 x 2*k+1
        A=A/sum(sum(A,1),2);
%         figure
%         surfA=surf(A);
%         set(surfA, 'edgecolor','none')
%         pause(0.5)
    else   % if the mask length is not an integer
        disp('Need to write the code!')
        A=[];
        return
        %         new_k=floor(k);
        %         extra = k-new_k;
        %         c=(2*m+1)/(2*new_k+1+2*extra);
        %
        %         a=zeros(1,2*new_k+3);
        %
        %         t=extra*c+1;
        %         t1=t-floor(t);
        %         %t2=ceil(t)-t;
        %         if k<0
        %             disp('Ops')
        %             a=[];
        %             return
        %         end
        %         a(1)=sum(w(1:floor(t)))+t1*w(floor(t));
        %
        %         for i=2:2*new_k+2
        %             s=extra*c+(i-2)*c+1;
        %             t=extra*c+(i-1)*c;
        %             %s1=s-floor(s);
        %             s2=ceil(s)-s;
        %
        %             t1=t-floor(t);
        %
        %
        %             a(i)=sum(w(ceil(s):floor(t)))+s2*w(ceil(s))+t1*w(floor(t));
        %         end
        %         t2=ceil(t)-t;
        %
        %         a(2*new_k+3)=sum(w(ceil(t):n))+t2*w(ceil(t));
    end
else % We need a filter with more points than MM, we use interpolation
    disp('Need to write the code!')
    A=[];
    return
    %     dx=0.01;
    %     % we assume that MM has a dx = 0.01, if m = 6200 it correspond to a
    %     % filter of length 62*2 in the physical space
    %     f=w/dx; % function we need to interpolate
    %     dy=m*dx/k;
    %     b=interp1(0:m,f(m+1:2*m+1),0:m/k:m);
    %     if size(b,1)>size(b,2)
    %         b=b.';
    %     end
    %     if size(b,1)>1
    %         fprintf('\n\nError!')
    %         disp('The provided mask is not a vector!!')
    %         a=[];
    %         return
    %     end
    %     a=[fliplr(b(2:end)) b]*dy;
    %     if abs(norm(a,1)-1)>10^-14
    %         %         fprintf('\n\nError\n\n')
    %         %         fprintf('Area under the mask = %2.20f\n',norm(a,1))
    %         %         fprintf('it should be equal to 1\nWe rescale it using its norm 1\n\n')
    %         a=a/norm(a,1);
    %     end
end

end




function maxmins = Maxmins(f,extensionType)

if nargin == 1, extensionType = 'c'; end
N = length(f);
maxmins=zeros(1,N);
df = diff(f);


h = 1;
cIn=0;
if strcmp(extensionType,'p') && df(1) == 0 && df(end) == 0
    while df(h)==0
        cIn=cIn+1;
        h=h+1;
    end
end

c = 0;
cmaxmins=0;
for i=h:N-2
    if   df(i)*df(i+1) <= 0
        if df(i+1) == 0
            if c == 0
                posc = i;
            end
            c = c + 1;
        else
            if c > 0
                cmaxmins=cmaxmins+1;
                maxmins(cmaxmins)=posc+floor((c-1)/2)+1;
                c = 0;
            else
                cmaxmins=cmaxmins+1;
                maxmins(cmaxmins)=i+1;
            end
        end
    end
end
if c > 0
    cmaxmins=cmaxmins+1;
    maxmins(cmaxmins)=mod(posc+floor((c+cIn-1)/2)+1,N);
    if maxmins(cmaxmins)==0
        maxmins(cmaxmins)=N;
    end
end

maxmins=maxmins(1:cmaxmins);

if strcmp(extensionType,'p') % we deal with a periodic signal
    if isempty(maxmins)
        maxmins = 1;
    else
        if maxmins(1)~=1 && maxmins(end)~=N
            if (f(maxmins(end)) > f(maxmins(end)+1) && f(maxmins(1)) > f(maxmins(1)-1)) || (f(maxmins(end)) < f(maxmins(end)+1) && f(maxmins(1)) < f(maxmins(1)-1))
                maxmins=[1 maxmins];
            end
        end
    end
elseif strcmp(extensionType,'c')
    if isempty(maxmins)
        maxmins = [1, N];
    else
        if maxmins(1) ~= f(1) && maxmins(end) ~= f(end)
            maxmins = [1, maxmins, N];
        elseif f(maxmins(1)) ~= f(1)
            maxmins = [1, maxmins];
        elseif  f(maxmins(end)) ~= f(end)
            maxmins = [maxmins, N];
        end
    end
elseif strcmp(extensionType,'r')
    if isempty(maxmins)
        maxmins = [1, N];
    else
        if maxmins(1) ~= f(1) && maxmins(end) ~= f(end)
            maxmins = [1, maxmins, N];
        elseif f(maxmins(1)) ~= f(1)
            maxmins = [1, maxmins];
        elseif  f(maxmins(end)) ~= f(end)
            maxmins = [maxmins, N];
        end
    end
end

end
