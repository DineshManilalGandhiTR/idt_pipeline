function success = output(acc, sensitivity, specificity, precision, confusion_mat, WF,RL,WL)

 
        
        
        xlwrite('./data/confusion_matrix.xlsx',confusion_mat,1);
        
        range1=sprintf('B%d',size(confusion_mat,1)+2);
        range2=sprintf('E%d',size(confusion_mat,1)+2);
        temp={'accuracy(TP+TN/P+N)','sensitivity(TP/P)','specificity(TN/N)','precision(TP/TP+FP)'};
        xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
        
        range1=sprintf('B%d',size(confusion_mat,1)+3);
        range2=sprintf('B%d',size(confusion_mat,1)+3+numel(acc)-1);
        %xlwrite('./data/confusion_matrix.xlsx',acc./numel(all_test_labels),[range1,':',range2]);
        xlwrite('./data/confusion_matrix.xlsx',acc,[range1,':',range2]);
        
        range1=sprintf('C%d',size(confusion_mat,1)+3);
        range2=sprintf('C%d',size(confusion_mat,1)+3+numel(sensitivity)-1);
        %xlwrite('./data/confusion_matrix.xlsx',sensitivity./size(fvt_pos_test,2),[range1,':',range2]);
        xlwrite('./data/confusion_matrix.xlsx',sensitivity,[range1,':',range2]);
        
        range1=sprintf('D%d',size(confusion_mat,1)+3);
        range2=sprintf('D%d',size(confusion_mat,1)+3+numel(specificity)-1);
        %xlwrite('./data/confusion_matrix.xlsx',specificity./size(fvt_neg_test,2),[range1,':',range2]);
        xlwrite('./data/confusion_matrix.xlsx',specificity,[range1,':',range2]);
        
        range1=sprintf('E%d',size(confusion_mat,1)+3);
        range2=sprintf('E%d',size(confusion_mat,1)+3+numel(precision)-1);
        xlwrite('./data/confusion_matrix.xlsx',precision,[range1,':',range2]);
        
        range1=sprintf('A%d',size(confusion_mat,1)+size(acc,1)+3);
        range2=sprintf('E%d',size(confusion_mat,1)+size(acc,1)+3);
        temp={'mean',mean(acc,'omitnan'),mean(sensitivity,'omitnan'),mean(specificity,'omitnan'),mean(precision,'omitnan')};
       % temp={'mean',acc,sensitivity,specificity,precision};
        xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
        
        range1=sprintf('B%d',size(confusion_mat,1)+size(acc,1)+4);
        range2=sprintf('B%d',size(confusion_mat,1)+size(acc,1)+4);
        xlwrite('./data/confusion_matrix.xlsx',trace(confusion_mat)/sum(sum(confusion_mat(2:end,2:end))),[range1,':',range2]);
        

        range1=sprintf('A%d',size(confusion_mat,1)+2+size(sensitivity,1)+4);
        range2=sprintf('C%d',size(confusion_mat,1)+2+size(sensitivity,1)+4);
        temp={'incorrectly classified videos','actual label','predicted label'};
        xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
        range1=sprintf('A%d',size(confusion_mat,1)+2+size(sensitivity,1)+5);
        range2=sprintf('A%d',size(confusion_mat,1)+2+size(sensitivity,1)+4+numel(WF));
        xlwrite('./data/confusion_matrix.xlsx',WF,[range1,':',range2]);
        range1=sprintf('B%d',size(confusion_mat,1)+2+size(sensitivity,1)+5);
        range2=sprintf('C%d',size(confusion_mat,1)+2+size(sensitivity,1)+4+numel(RL));
        xlwrite('./data/confusion_matrix.xlsx',[RL,WL],[range1,':',range2]);
        
        success=1;