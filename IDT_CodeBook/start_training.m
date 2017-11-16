%cd where the codes locate
cd /home/IDTcodebook/

run('./vlfeat-0.9.20/toolbox/vl_setup')
javaaddpath('./20130227_xlwrite/poi_library/dom4j-1.6.1.jar');
javaaddpath('./20130227_xlwrite/poi_library/poi-3.8-20120326.jar');
javaaddpath('./20130227_xlwrite/poi_library/poi-ooxml-3.8-20120326.jar');
javaaddpath('./20130227_xlwrite/poi_library/poi-ooxml-schemas-3.8-20120326.jar');
javaaddpath('./20130227_xlwrite/poi_library/xmlbeans-2.3.0.jar');
path=pwd;
addpath(path);
addpath(fullfile(path,'20130227_xlwrite'));
addpath(fullfile(path,'libsvm'));
addpath(fullfile(path,'libsvm/matlab'));
addpath(fullfile(path,'vgg_fisher'));
addpath(fullfile(path,'vgg_fisher/lib/gmm-fisher/matlab'));

% set path for train and test list
params_training.train_file_dir=fullfile(path,'lists/split/train_test_list/'); 
params_training.test_file_dir=fullfile(path,'lists/split/train_test_list/');

% set path for train and test feature data
params_training.dtf_dir='./data/idt_feature/training/';
params_testing.dtf_dir='./data/idt_feature/testing/';

params_training.K=256; % num of GMMs
params_training.DTF_subsample_num=1000; % Subsampling number of DTF features per video clip

pred_results=fullfile(path,'data',sprintf('pred_results_sample%d_gmm%d.mat',params_training.DTF_subsample_num,params_training.K));

% load codebook
load('./data/codebook/pca_gmm_data_train_sample1000_gmm256.mat');

all_train_labels=[];
all_test_labels=[];
all_train_files=[];
all_test_files=[];

%set pattern for train and test list file
reg_pattern_train='trainlist.txt*';
reg_pattern_test='testlist.txt*';

train_lists=dir(fullfile(params_training.train_file_dir,reg_pattern_train));
test_lists=dir(fullfile(params_training.test_file_dir,reg_pattern_test));

% read all train files
for i=1:length(train_lists)
    fid=fopen(fullfile(params_training.train_file_dir,train_lists(i).name));
    tmp=textscan(fid,'%s%d');
    all_train_files=[all_train_files;tmp{1}];
    all_train_labels=[all_train_labels;tmp{2}];
end

% wipe off the list of non-exist or empty files
counter = 1;
while 1
    if counter == 0 
        counter = 1;
    else if counter > length(all_train_files)
        break 
        end
    end
    train_file = all_train_files(counter);
    train_file_str = train_file{1};
    tmp_train_file= fullfile(params_training.dtf_dir, strcat(train_file_str, '_iDT_Features.bin'));
    fileInfo = dir(tmp_train_file);
    if  ~exist(tmp_train_file) | fileInfo.bytes < 1024
        all_train_files(counter) = [];
        all_train_labels(counter) = [];
        counter = counter - 1;
    else
        counter = counter + 1;
    end  
end
      
% read all test files
for i=1:length(test_lists)
    fid=fopen(fullfile(params_training.test_file_dir,test_lists(i).name));
    tmp=textscan(fid,'%s%d');
    all_test_files=[all_test_files;tmp{1}];
    all_test_labels=[all_test_labels;tmp{2}];
end

% wipe off the list of non-exist or empty files
counter = 1;
while 1
    if counter == 0
        counter = 1;
    else if counter > length(all_test_files)
        break
        end
    end
    test_file = all_test_files(counter);
    test_file_str = test_file{1};
    tmp_test_file= fullfile(params_testing.dtf_dir, strcat(test_file_str, '_iDT_Features.bin'));
    fileInfo = dir(tmp_test_file);
    if  ~exist(tmp_test_file) | fileInfo.bytes < 1024
        all_test_files(counter) = [];
        all_test_labels(counter) = [];
        counter = counter - 1;
    else
        counter = counter + 1;
    end    
end


% Load trainning videos, computer Fisher vectors and train SVM model
svm_option='-t 0 -s 0 -q -w0 0.5 -w1 0.5 -c 100 -b 1'; % temporarily not to use linear SVM
uniq_labels=unique(all_train_labels);
pred=zeros(length(all_test_files),1);
acc=zeros(numel(uniq_labels),1);
sensitivity=zeros(numel(uniq_labels),1);
specificity=zeros(numel(uniq_labels),1);
precision=zeros(numel(uniq_labels),1);
result=zeros(length(all_test_files),numel(uniq_labels));



for i=1:numel(uniq_labels)
    tic;
    fileName=sprintf('fisher_vectors_action%d_sample%d_gmm%d.mat',i,params_training.DTF_subsample_num,params_training.K);
    classifiersName=sprintf('svm_model_action%d.mat',i);
    if ~exist(fullfile(path,'data',classifiersName),'file')|| ~exist(fullfile(path,'data',fileName),'file')
        % Process training files
        disp('Train begins...')
        pos_idx=(all_train_labels == i);
        pos_files=all_train_files(pos_idx); % positive training files
        % load(fullfile(path,'data',fileName));
        fvt_pos_train = compute_fvt(params_training, pca_coeff, gmm, pos_files,'training');
        % fvt_pos_train=compute_fisher(params, pca_coeff, gmm, pos_files);
        % ENC = VL_FISHER(X, means, covariances, priors);
        
        neg_idx=(all_train_labels ~= i);
        neg_files=all_train_files(neg_idx);
        sample_idx=randperm(length(neg_files),min(length(neg_files),length(pos_files)));
        sample_neg_files=neg_files(sample_idx);
        fvt_neg_train = compute_fvt(params_training, pca_coeff, gmm, sample_neg_files,'training');
        
        % Train SVM model
 
        tmp_train_labels=[ones(1,size(fvt_pos_train,2)) -1*ones(1,size(fvt_neg_train,2))];
        
        if ~exist(fullfile(path,'data',classifiersName),'file')
            model=svmtrain(double(tmp_train_labels)', double([fvt_pos_train fvt_neg_train])', svm_option);
            %model = fitcsvm(double([fvt_pos_train fvt_neg_train])',double(tmp_train_labels)','Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        else
            load(fullfile(path,'data',classifiersName));
        end
        disp('Train done!');
        % Process test files
        disp('Test begins...');
        pos_idx=(all_test_labels == i);
        pos_files=all_test_files(pos_idx); % positive testing files
        fvt_pos_test = compute_fvt(params_testing, pca_coeff, gmm, pos_files,'training');
        
        neg_idx=(all_test_labels ~= i);
        neg_files=all_test_files(neg_idx);
        % sample_idx=randperm(length(neg_files),min(length(neg_files),length(pos_files)));
        sample_idx=randperm(length(neg_files),length(neg_files));
        sample_neg_files=neg_files(sample_idx);
        fvt_neg_test = compute_fvt(params_testing, pca_coeff, gmm, sample_neg_files,'training');
        
        size_pos_test(i)=size(fvt_pos_test,2);
        size_neg_test(i)=size(fvt_neg_test,2);
        
        % SVM prediction
        tmp_test_labels=[ones(1,size(fvt_pos_test,2)) -1*ones(1,size(fvt_neg_test,2))];
       
        [pred_labels,accuracy,prob_estimates] = svmpredict(double(tmp_test_labels)', double([fvt_pos_test fvt_neg_test])', model, '-b 1');
        %[pred_labels, prob_estimates] = predict(model, double([fvt_pos_test fvt_neg_test])');
        result(find(pos_idx==1),i)=prob_estimates(1:size(fvt_pos_test,2),1);
        result(find(pos_idx~=1),i)=prob_estimates(size(fvt_pos_test,2)+1:size(fvt_neg_test,2)+size(fvt_pos_test,2),1);
        
        % acc(i) = (sum(pred_labels(1:size(fvt_pos_test,2)) == 1)+sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2)) ~=1)) ./ (size(fvt_pos_test,2)+size(fvt_neg_test,2))    %# accuracy
        % sensitivity(i) = sum(pred_labels(1:size(fvt_pos_test,2)) == 1) ./ size(fvt_pos_test,2);
        % specificity(i) = sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2)) ~=1)./size(fvt_neg_test,2);
        % precision(i) = sum(pred_labels(1:size(fvt_pos_test,2)) == 1)./(sum(pred_labels(1:size(fvt_pos_test,2)) == 1)+sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2))==1));
        
        [acc(i), sensitivity(i), specificity(i), precision(i)] = compute_binary_class_index(pred_labels, size(fvt_pos_test,2), size(fvt_neg_test,2))
        pred(pos_idx) = int32(pred_labels(1:size(fvt_pos_test,2))==1)*uniq_labels(i);
        % pred = [pred; (pred_labels(1:size(fvt_pos_test,2))==1)*i];
        disp('Test done!');
        save_file=sprintf('fisher_vectors_action%d_sample%d_gmm%d.mat',i,params_training.DTF_subsample_num,params_training.K);
        save(fullfile(path,'data',save_file),'fvt_pos_train','fvt_neg_train', 'fvt_pos_test', 'fvt_neg_test', '-v7.3');
        save_model=sprintf('svm_model_action%d.mat',i);
        save(fullfile(path,'data',save_model),'model','-v7.3');
    else
        % Process test files
        disp('Test begins...');
        load(fullfile(path,'data',fileName));
        load(fullfile(path,'data',classifiersName));
        pos_idx=(all_test_labels == i);
        
        % pos_files=all_test_files(pos_idx); % positive training files
        % fvt_pos_test = compute_fvt(params, pca_coeff, gmm, pos_files);
        % neg_idx=(all_test_labels ~= i);
        % neg_files=all_test_files(neg_idx);
        % sample_idx=randperm(length(neg_files),length(pos_files));
        % sample_neg_files=neg_files(sample_idx);
        % fvt_neg_test = compute_fvt(params, pca_coeff, gmm, sample_neg_files);
        
        size_pos_test(i)=size(fvt_pos_test,2);
        size_neg_test(i)=size(fvt_neg_test,2);
        
        tmp_test_labels=[ones(1,size(fvt_pos_test,2)) -1*ones(1,size(fvt_neg_test,2))];
        [pred_labels,accuracy,prob_estimates] = svmpredict(double(tmp_test_labels)', double([fvt_pos_test fvt_neg_test])', model, '-b 1');
        %[pred_labels, prob_estimates] = predict(model, double([fvt_pos_test fvt_neg_test])');
        result(find(pos_idx==1),i)=prob_estimates(1:size(fvt_pos_test,2),1);
        result(find(pos_idx~=1),i)=prob_estimates(size(fvt_pos_test,2)+1:size(fvt_neg_test,2)+size(fvt_pos_test,2),1);
        
        % acc(i) = (sum(pred_labels(1:size(fvt_pos_test,2)) == 1)+sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2)) ~=1)) ./ (size(fvt_pos_test,2)+size(fvt_neg_test,2))    %# accuracy
        % sensitivity(i) = sum(pred_labels(1:size(fvt_pos_test,2)) == 1) ./ size(fvt_pos_test,2);
        % specificity(i) = sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2)) ~=1)./size(fvt_neg_test,2);
        % precision(i) = sum(pred_labels(1:size(fvt_pos_test,2)) == 1)./(sum(pred_labels(1:size(fvt_pos_test,2)) == 1)+sum(pred_labels(size(fvt_pos_test,2)+1:size(fvt_pos_test,2)+size(fvt_neg_test,2))==1));
        [acc(i), sensitivity(i), specificity(i), precision(i)] = compute_binary_class_index(pred_labels, size(fvt_pos_test,2), size(fvt_neg_test,2))
        pred(pos_idx) = int32(pred_labels(1:size(fvt_pos_test,2))==1)*uniq_labels(i);
        disp('Test done!');
    end
    toc;
end


% [result_estimate,result_label]=max(result,[],2);
% confusion_mat=zeros(numel(uniq_labels)+1,numel(uniq_labels)+1);
% confusion_mat(1,:)=[0:1:numel(uniq_labels)];
% confusion_mat(:,1)=[0:1:numel(uniq_labels)];
% file_label=strings(size(result_label,1),1);

[confusion_mat,result_label]=compute_confusion_mat(result, uniq_labels,all_test_labels);

WF=all_test_files(find(all_test_labels~=result_label));
RL=all_test_labels(find(all_test_labels~=result_label));
WL=result_label(find(all_test_labels~=result_label));

% for actLa=1:numel(uniq_labels)
% %for actLa=sort(uniq_labels)'
%     for preLa = 1:numel(uniq_labels)
%     %for preLa = sort(uniq_labels)'
%         confusion_mat(actLa+1,preLa+1)=sum(result_label(find(all_test_labels==actLa))==preLa);
%     end
% end


s=output(acc, sensitivity, specificity, precision, confusion_mat, WF,RL,WL);
fprintf('\nMean accuracy: %f.\n', sum(pred==all_test_labels)/numel(all_test_labels)); % display mean accuracy 
fprintf('\nfinal accuracy: %f.\n', sum(result_label==all_test_labels)/numel(all_test_labels)); 

% %fprintf('Mean accuracy: %f.\n', mean(acc)); % display mean accuracy
% fprintf('\nMean accuracy: %f.\n', sum(pred==all_test_labels)/numel(all_test_labels)); % display mean accuracy
% fprintf('\nMean accuracy(revised): %f.\n', mean(acc./numel(all_test_labels)));
% fprintf('\nfinal accuracy: %f.\n', sum(result_label==all_test_labels)/numel(all_test_labels));
%
% %save(pred_results,'result_result','acc','-v7.3');
% xlwrite('./data/confusion_matrix.xlsx',confusion_mat,1);
%
% range1=sprintf('B%d',size(confusion_mat,1)+2);
% range2=sprintf('E%d',size(confusion_mat,1)+2);
% temp={'accuracy(TP+TN/P+N)','sensitivity(TP/P)','specificity(TN/N)','precision(TP/TP+FP)'};
% xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
%
% range1=sprintf('B%d',size(confusion_mat,1)+3);
% range2=sprintf('B%d',size(confusion_mat,1)+3+numel(acc)-1);
% xlwrite('./data/confusion_matrix.xlsx',acc./numel(all_test_labels),[range1,':',range2]);
%
% range1=sprintf('C%d',size(confusion_mat,1)+3);
% range2=sprintf('C%d',size(confusion_mat,1)+3+numel(sensitivity)-1);
% xlwrite('./data/confusion_matrix.xlsx',sensitivity,[range1,':',range2]);
%
% range1=sprintf('D%d',size(confusion_mat,1)+3);
% range2=sprintf('D%d',size(confusion_mat,1)+3+numel(specificity)-1);
% xlwrite('./data/confusion_matrix.xlsx',specificity,[range1,':',range2]);
%
% range1=sprintf('E%d',size(confusion_mat,1)+3);
% range2=sprintf('E%d',size(confusion_mat,1)+3+numel(precision)-1);
% xlwrite('./data/confusion_matrix.xlsx',precision,[range1,':',range2]);
%
% range1=sprintf('A%d',size(confusion_mat,1)+size(acc,1)+3);
% range2=sprintf('E%d',size(confusion_mat,1)+size(acc,1)+3);
% temp={'mean',mean(acc./numel(all_test_labels),'omitnan'),mean(sensitivity,'omitnan'),mean(specificity,'omitnan'),mean(precision,'omitnan')};
% xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
%
% %   range1=sprintf('A%d',size(confusion_mat,1)+2+size(sensitivity,1)+3);
% %   range2=sprintf('D%d',size(confusion_mat,1)+2+size(sensitivity,1)+3);
% %   temp={'video name','actual label','predicted label','type'};
% %   xlwrite('./data/confusion_matrix.xlsx',temp,[range1,':',range2]);
% %   range1=sprintf('A%d',size(confusion_mat,1)+2+size(sensitivity,1)+4);
% %   range2=sprintf('D%d',size(confusion_mat,1)+2+size(sensitivity,1)+3+size(all_labels,1));
% %   xlwrite('./data/confusion_matrix.xlsx',cellstr(all_labels),[range1,':',range2]);


try
	matlabpool close;
catch exception
end

fprintf('All done!\n');
