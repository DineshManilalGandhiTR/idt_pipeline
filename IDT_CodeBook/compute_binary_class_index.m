function [acc, sensitivity, specificity, precision] = compute_binary_class_index(pred_labels, pos_test_num, neg_test_num)

acc = (sum(pred_labels(1:pos_test_num) == 1)+sum(pred_labels(pos_test_num+1:pos_test_num+neg_test_num) ~=1)) ./ (pos_test_num+neg_test_num);    %# accuracy
sensitivity = sum(pred_labels(1:pos_test_num) == 1) ./ pos_test_num;
specificity = sum(pred_labels(pos_test_num+1:pos_test_num+neg_test_num) ~=1)./neg_test_num;
precision = sum(pred_labels(1:pos_test_num) == 1)./(sum(pred_labels(1:pos_test_num) == 1)+sum(pred_labels(pos_test_num+1:pos_test_num+neg_test_num)==1)); 