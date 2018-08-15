function [result] = f_CSP(varargin) 

    if (nargin ~= 2)
        disp('Must have 2 classes for CSP!')
    end
    
    Rsum=0;
    % finding the covariance of each class and composite covariance
    for i = 1:nargin 
        %mean here?
        R{i} = ((varargin{i}*varargin{i}')/trace(varargin{i}*varargin{i}'));
        % Ramoser equation 
        Rsum=Rsum+R{i};
    end
   
    %   Find Eigenvalues and Eigenvectors of RC
    %   Sort eigenvalues in descending order
    [EVecsum,EValsum] = eig(Rsum);
    [EValsum,ind] = sort(diag(EValsum),'descend');
    EVecsum = EVecsum(:,ind);
    
    %   Find Whitening Transformation Matrix - Ramoser Equation 
        W = sqrt(inv(diag(EValsum))) * EVecsum';
    
    for k = 1:nargin
        S{k} = W * R{k} * W'; % Whiten Data Using Whiting Transform 
    end
   
    % generalized eigenvectors/values
    [B,D] = eig(S{1},S{2});
    % Simultanous diagonalization
			% Should be equivalent to [B,D]=eig(S{1});
    
    [D,ind]=sort(diag(D));B=B(:,ind);
    
    %Resulting Projection Matrix-these are the spatial filter coefficients
    result = B'*W;
end
