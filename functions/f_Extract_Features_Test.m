function [TEST_DB]=f_Extract_Features_Test(test,time_point,Samp_Pts,Total_Trials,No_of_Components )


Test_Db=test(:,1:Samp_Pts);
for i=2:Total_Trials
    Temp=test(:,(i-1)*Samp_Pts+1:i*Samp_Pts);
    Test_Db=cat(3,Test_Db,Temp);
end

TEST=Test_Db(:,:,:);


for i=1:Total_Trials
    for j=1:No_of_Components
       Temp=var(TEST(j,:,i));
       TEST_DB(i,j)=log(Temp);    
    end
end