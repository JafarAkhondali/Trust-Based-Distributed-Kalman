% Trust-Based Distributed Kalman - Replay Attack

% C. Liang, F. Wen, and Z. Wang, "Trust-Based Distributed Kalman Filtering for Target Tracking under Malicious Cyber Attacks", Information Fusion, vol. 46, pp.44-50, Mar. 2019

% https://doi.org/10.1016/j.inffus.2018.04.002

% Author: Fuxi Wen, Date: 2019-03-06

clear
close all
clc

tic
% Make a point move in the 2D plane
% State = (x y xdot ydot). We only observe (x y).

% This code was used to generate Figure 15.9 of "Artificial Intelligence: a Modern Approach",
% Russell and Norvig, 2nd edition, Prentice Hall, 2003.

% X(t+1) = F X(t) + noise(Q)
% Y(t) = H X(t) + noise(R)

p_nodes = 7; % number of nodes

p_num = 20; % number of measurements

labelEst = zeros(p_num,p_nodes);
X = zeros(2,p_nodes);

p_state = 4; % state size
p_observation = 2; % observation size

F = [1 0 1 0; ...
    0 1 0 1; ...
    0 0 1 0; ...
    0 0 0 1]; 

H = [1 0 0 0; ...
    0 1 0 0];

Q =  1*eye(p_state); % variance of states

initx = [10 10 1 0]';
initV = 10*eye(p_state);

% % -------------------------------------------------------------------------
% %                    Different SNRs for different nodes
% % -------------------------------------------------------------------------
p_noisy_num = 0; % number of noisy nodes
% 
snrS = [10*ones(1,p_noisy_num) 10*ones(1,p_nodes-p_noisy_num)] + 0.5*rand(1,p_nodes);
% 
% % 0: noisy nodes and 1: high SNR nodes
% labelTrue = ones(p_num,1)*[zeros(1,p_noisy_num) ones(1,p_nodes-p_noisy_num)];

% -------------------------------------------------------------------------
%                Nodes are under attack or without attack
% -------------------------------------------------------------------------
p_unTrust_num = 3;

% 1: under attack, 0: no attack
%     Boolean = zeros(1,p_nodes); %  
    Boolean = [ones(1,p_unTrust_num) zeros(1,p_nodes-p_unTrust_num)];  

% 0: without attack 1: FDI attack, 2: Reply attack, 3: Combined attack
p_type = 2; % or 2

switch p_type
    case 1
        disp('FDI Attack')
        type = [p_type*ones(1,p_unTrust_num) zeros(1,p_nodes-p_unTrust_num)];
    case 2
        disp('Replay Attack')
        type = [p_type*ones(1,p_unTrust_num) zeros(1,p_nodes-p_unTrust_num)];
    case 3
        disp('Combined Attack')
        type = [randi(2,1,p_unTrust_num) zeros(1,p_nodes-p_unTrust_num)]; 
    otherwise
        disp('Unknown method.')
        type = zeros(1,p_nodes);
end


delayL = 3;
delayS = -1;
delayH = 1;

delayArray = delayL:delayS:delayH;

rmse_smooth_It = zeros(length(delayArray),p_num); % Uniform weights
rmse_smooth2_It = zeros(length(delayArray),p_num); % Uniform weights
rmse_smooth3_It = zeros(length(delayArray),p_num); % Uniform weights

for itd  = 1:length(delayArray)

% 1. parameter of FDI attack
    attackMean = 1*ones(1,p_nodes).*Boolean; % amplitude for FDI attack
    attackStdDev = 1*ones(1,p_nodes).*Boolean; % amplitude for FDI attack
    
% 2. parameter of Reply attack
    delay = delayArray(itd)*ones(1,p_nodes).*Boolean;% for replay attack
% -------------------------------------------------------------------------

% -----------------------
runs = 50;
% -----------------------

pd = zeros(1,runs);
rmse_smooth_temp = zeros(runs,p_num);
rmse_smooth2_temp = zeros(runs,p_num);
rmse_smooth3_temp = zeros(runs,p_num);

for it = 1:runs
    
    str_node = struct('data',[]);
    
    Xmean = zeros(2,p_num);
    Xweight = zeros(2,p_num);
    Xnew = zeros(2,p_num);
    
    dX = zeros(p_nodes,p_num);
    
    for node = 1:p_nodes
    
        str_node(node).snr = snrS(node); % signal to noise ratio (snr)
    
        str_node(node).R = 10^(-snrS(node)/10)*eye(p_observation); % noise variance
    
        str_node(node).TF = Boolean(node); % 1: with or 0: without attack (trust)
        str_node(node).type = type(node); % 1: with or 0: without attack (trust)
        
        % generate the states and observations
        s = rng(it);
    
        [str_node(node).x,str_node(node).y] = sample_lds(F, H, Q, str_node(node).R, initx, p_num);
    
        
        % Kalman filtering and smoothing
        [str_node(node).xsmooth, str_node(node).Vsmooth] = kalman_smoother(str_node(node).y, F, H, Q, str_node(node).R, initx, initV);
        
        switch str_node(node).type
          case 1
%             disp('FDI Attack')
            
            attackVec = attackMean(node)*ones(size(str_node(node).xsmooth)) + ...
                attackStdDev(node)*randn(size(str_node(node).xsmooth));
            str_node(node).xsmooth = str_node(node).xsmooth + attackVec;
            
          case 2
%             disp('Reply Attack')
            str_node(node).xsmooth = [zeros(p_state,delay(node)) str_node(node).xsmooth(:,1:end-delay(node))];
            
          otherwise
%             disp('Nordelay(node)mal.')
        end
        rng(s); 
    end

    % ---------------------------  Algorithm 1 ----------------------------
    % Compute RMSE: uniform (Xall)
    for node = 1:p_nodes
        Xmean = Xmean + str_node(node).xsmooth([1,2],:)/p_nodes;
    end
    
    % uniform weights
    dsmooth = (str_node(node).x([1 2],:) - Xmean);%./str_node(node).x([1 2],:) ; 
    rmse_smooth_temp(it,:) = sqrt(sum(dsmooth.^2));
    
    % ---------------------------  Algorithm 2 ----------------------------
    % Compute RMSE: weighting (Xall)
    for node = 1:p_nodes
        for itn = 1:p_num
            dX(node,itn) = 1/norm(str_node(node).xsmooth([1,2],itn) - Xmean(:,itn));
        end
    end
    wX = dX./(ones(p_nodes,1)*sum(dX)); % weighting matrix
    
%     Xweight = zeros(2,p_num);
    for itn = 1:p_num
        for node = 1:p_nodes
            Xweight(:,itn) = Xweight(:,itn) + wX(node,itn)*str_node(node).xsmooth([1,2],itn);
        end
    end
    
    % 
    dsmooth2 = (str_node(node).x([1 2],:) - Xweight);%./str_node(node).x([1 2],:) ; 
    rmse_smooth2_temp(it,:) = sqrt(sum(dsmooth2.^2));
    
    % -------------------------- Proposed Trust-based ---------------------
    % clustering - k-means
    for itn = 1:p_num
        for node = 1:p_nodes
            X(:,node) = str_node(node).xsmooth([1,2],itn);
        end
        k = 2; % number of the clusters
        [idx,C] = kmeans(X,k); % k-means clustering
        uniqueVal = unique(idx); % find the unique cluster label
        countVal = histc(idx,uniqueVal); % count the members in each cluster
        [Y,Index] = max(countVal); % find the cluster with largest members
        trust = uniqueVal(Index); % find the Trust nodes
        labelEst(itn,:) = double(idx==trust); % trust: 1 and not trust: 0
        Xnew(1:2,itn) = mean(X(1:2,idx==trust),2); % remove the measurements from un-Trust nodes
    end

    % Compute RMSE: trust-based weights (Xnew)
    dsmooth3 = (str_node(node).x([1 2],:) - Xnew);%./str_node(node).x([1 2],:) ; 
    rmse_smooth3_temp(it,:) = sqrt(sum(dsmooth3.^2));
    
%     % detection performance
%     detct = abs(sum(labelTrue - labelEst,2));
%     pd(1,it) = sum(double(detct==0))/p_num; % probability of detection
    % ---------------------------------------------------------------------
end

rmse_smooth_It(itd,:) = mean(rmse_smooth_temp); % Uniform weights
rmse_smooth2_It(itd,:) = mean(rmse_smooth2_temp); % Variance weights
rmse_smooth3_It(itd,:) = mean(rmse_smooth3_temp); % Trust-based

end
% ----------------------- Figure -----------------------

rmse_IT = [rmse_smooth_It;rmse_smooth2_It;rmse_smooth3_It];

figure(1)
% Starting in R2014b
ColorSpec = {[ 0    0.4470    0.7410],...
    [ 0.8500    0.3250    0.0980],...
    [ 0.9290    0.6940    0.1250],...
    [ 0.4940    0.1840    0.5560],...
    [ 0.4660    0.6740    0.1880],...
    [ 0.3010    0.7450    0.9330],...
    [ 0.6350    0.0780    0.1840]};

LineStyle = {':o',':*',':<','-o','.--','-v','-','x','s'};

strP = cell(1,size(rmse_IT,1));
px = max(delayArray)+1:p_num; 

p_delay = [delayArray delayArray delayArray];

index = 0;
for it = 1:size(rmse_IT,1)
        index = index + 1;
        
        % LineStyle
        if mod(index,length(LineStyle))
            itL = mod(index,length(LineStyle));
        else
            itL = length(LineStyle);
        end
        % ColorSpec
        if mod(index,length(ColorSpec))
            itC = mod(index,length(ColorSpec));
        else
            itC = length(ColorSpec);
        end
        
        plot(px,rmse_IT(it,max(delayArray)+1:p_num),LineStyle{itL},'Color',ColorSpec{itC},'LineWidth',2);
        
        if it<size(delayArray,2)+1
            strP{index}=['Uniform [55]: \tau = ',num2str(p_delay(it))];
        elseif it <2*size(delayArray,2)+1
            strP{index}=['Relative [56]: \tau = ',num2str(p_delay(it))];
        elseif 2*size(delayArray,2) < it
            strP{index}=['Trust: \tau = ',num2str(p_delay(it))];
        end

        hold on
 
end

grid

legend(strP)

xlabel('time','fontsize',12)
ylabel('Instantaneous RMSE','fontsize',12)
hold off

% ----------------------- Figure -----------------------
figure(2) 
% % R2014a and Earlier
ColorSpec = {[  0         0    1.0000],...
         [0    0.5000         0],...
    [1.0000         0         0],...
     [    0    0.7500    0.7500],...
    [0.7500         0    0.7500],...
    [0.7500    0.7500         0],...
    [0.2500    0.2500    0.2500]};

% Starting in R2014b
% ColorSpec = {[ 0    0.4470    0.7410],...
%     [ 0.8500    0.3250    0.0980],...
%     [ 0.9290    0.6940    0.1250],...
%     [ 0.4940    0.1840    0.5560],...
%     [ 0.4660    0.6740    0.1880],...
%     [ 0.3010    0.7450    0.9330],...
%     [ 0.6350    0.0780    0.1840]};

LineStyle = {'o','d','+','*','.','x','^','>'};

p_nodes = 0;

strP = cell(1,p_nodes);

plot(str_node(1).x(1,:),str_node(1).x(2,:),'k-',Xnew(1,:),Xnew(2,:),'rs','LineWidth',2)

strP{p_nodes+1} = 'True Trajectory';
strP{p_nodes+2} = 'Trust-based Trajectory';
grid
legend(strP)

xlabel('x','fontsize',12)
ylabel('y','fontsize',12)
hold off

toc