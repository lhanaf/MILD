function evaluation(flag_file, probability_file, truth_file)

fp = fopen(probability_file,'rb');
ori_p = fread(fp,'float');
fclose(fp);
fp = fopen(flag_file,'rb');
ori_f = fread(fp,'float');
fclose(fp);

frame_num = floor(sqrt(length(ori_f)));
input_probability = zeros(frame_num,frame_num);
input_flag = zeros(frame_num,frame_num);
visited_probability = zeros(frame_num,frame_num);
visited_flag = zeros(frame_num,frame_num);


try 
    load(truth_file);
catch 
    fprintf(1,'no truth files\r\n');
    truth = zeros(frame_num,frame_num);
end

for i = 1:frame_num
    input_probability(i,:) = ori_p(i*frame_num-frame_num+1:i*frame_num);
    input_flag(i,:) = ori_f(i*frame_num-frame_num+1:i*frame_num);
end

visited_flag = input_flag;

truth = truth > 0;
tp = (visited_flag == truth & truth > 0);
detected_true_positive = sum(tp,2) > 0;
true_positive = sum(truth,2) >0;
detected_positive = sum(visited_flag,2) > 0;
recall = sum(detected_true_positive == true_positive & true_positive > 0) / sum(true_positive);
precision = sum(detected_positive == true_positive & true_positive > 0) / sum(detected_positive);
if(1)
    figure;
    imagesc((visited_flag - truth) * 2 + truth  );
    figure;
    plot(true_positive,'r*');
    hold on; 
    plot(detected_positive,'b');
    legend('true positive', 'detected positive');
    fprintf(1,'recall : %f\n', recall);
    fprintf(1,'precision : %f\n', precision);
    figure;imagesc(input_probability);title('input probability');
    figure;imagesc(input_flag);title('input flag');

end
endfunction
