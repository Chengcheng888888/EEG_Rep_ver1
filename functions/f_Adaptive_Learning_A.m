
%                    Function Name:f_Adaptive_Learning_A


function [LABEL]=f_Adaptive_Learning_A(TEST_X,TR_MDL)

    [No_of_Trails, Dim]=size(TEST_X);

% For each observation compute the UCl and LCL
    for i=1:No_of_Trails;                  
         input_for_pred=TEST_X(i,:); 
         [label]=f_BCI_Classification(input_for_pred,TR_MDL); % Classification Function                  
         LABEL.SVM(i)=label.SVM;

    end
end
