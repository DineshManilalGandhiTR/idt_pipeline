function [confusion_mat,result_label]=compute_confusion_mat(prob_result, uniq_labels,all_test_labels)

[result_estimate,result_label]=max(prob_result,[],2);
confusion_mat=zeros(numel(uniq_labels)+1,numel(uniq_labels)+1);
confusion_mat(1,2:end)=uniq_labels;
confusion_mat(2:end,1)=uniq_labels;

for actLa=1:numel(uniq_labels)
%for actLa=sort(uniq_labels)'
    for preLa = 1:numel(uniq_labels)
    %for preLa = sort(uniq_labels)'
        confusion_mat(actLa+1,preLa+1)=sum(result_label(find(all_test_labels==actLa))==preLa);
    end
end