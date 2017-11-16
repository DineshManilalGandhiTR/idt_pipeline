% Script to test DTF features and Fisher Vector
% cd where the code locates
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

%% Set Parameters
params.K=256;   % num of GMMs
params.DTF_subsample_num=1000; % Subsampling number of DTF features per video clip

params.encoding='fisher'; % encoding type: 'fisher' - fisher vector; 'bow' - bag-of-words
params.feat_list={'Trajectory','HOG','HOF','MBHx','MBHy'}; % all features involved in this test
feat_len={30,96,108,96,96}; % length of features
params.feat_len_map=containers.Map(params.feat_list, feat_len);
params.feat_start=41; % start position of DTF features, the first 40 are just some describe info

% params.dtf_dir=fullfile(path,'Possible actions');% UCF101_DTF,violentflow
% params.train_list_dir=fullfile(path, 'Possible actions'); %
% params.test_list_dir=fullfile(path, 'Possible actions'); 
params.dtf_dir='./data/idt_feature/pretrain/'; % idt feature file for pretrain 
params.train_list_dir=fullfile(path,'lists/split/train_test_list/'); % lists for pretrain
params.test_list_dir=fullfile(path,'lists/split/train_test_list');

params.train_data_info=fullfile(path,'data','PA_traindata_info.mat'); % the folder 'data' will hold the mat codebook
params.test_data_info=fullfile(path,'data','PA_testdata_info.mat');

% Files to store subsampled features
params.train_sample_data=fullfile(path,'data',sprintf('PA_train_data_sample%d_gmm%d.mat',params.DTF_subsample_num,params.K));
params.test_sample_data=fullfile(path,'data',sprintf('PA_test_data_sample%d_gmm%d.mat',params.DTF_subsample_num,params.K));

% Files to store Fisher vectors
params.fv_train_file=fullfile(path,'data',sprintf('pca_gmm_data_train_sample%d_gmm%d.mat',params.DTF_subsample_num,params.K));
params.fv_test_file=fullfile(path,'data',sprintf('pca_gmm_data_test_sample%d_gmm%d.mat',params.DTF_subsample_num,params.K));



try
	matlabpool close;
catch exception
end
% matlabpool open 4
fvt=0;

        %% Construct Fisher Vectors for training
        % Subsample DTF features, calculate PCA coefficients and train GMM model
        if ~exist(params.fv_train_file,'file') && ~exist(params.fv_test_file,'file')
            % Train and test share the same codebook?
            [ pca_coeff, gmm, all_train_files_codebook, all_train_labels_codebook, all_test_files_codebook, all_test_labels_codebook ] = pretrain(params);
            save(params.fv_train_file,'pca_coeff','gmm','all_train_files_codebook','all_train_labels_codebook','all_test_files_codebook', 'all_test_labels_codebook', '-v7.3');
        else
            load(params.fv_train_file); 
        end
        
