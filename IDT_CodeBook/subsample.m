function [X,file_list,labels] = subsample(params,t_type)
% SUBSAMPLE subsample DTF features
% outputs:
%	X - cell of DTF features
%	labels - labels of corresponding DTF features
%	file_list - list of all files

X=cell(length(params.feat_list),1);%length(params.feat_list)
labels=[];

switch t_type
    case 'train'
        tt_list_dir=params.train_list_dir;
        reg_pattern='trainlist*'; % name pattern of trainlist
    case 'test'
        tt_list_dir=params.test_list_dir;
        reg_pattern='testlist*'; % name pattern of testlist
    otherwise
        error('Unknown file pattern!');
end

% extract training/test list
tt_list=[]; % train/test files
tlists=dir(fullfile(tt_list_dir,reg_pattern));
for i=1:length(tlists)
    fid=fopen(fullfile(tt_list_dir,tlists(i).name));
    tmp=textscan(fid,'%s%d');
    tt_list=[tt_list;tmp{1}];
    labels=[labels;tmp{2}];
end

%eliminate all empty or non-exist files
counter = 1;
while 1
    if counter == 0 
        counter = 1;
    else if counter > length(tt_list)
        break 
        end
    end
    train_file = tt_list(counter);
    train_file_str = train_file{1};
    tmp_train_file= fullfile(params.dtf_dir, strcat(train_file_str, '_iDT_Features.bin'));
    fileInfo = dir(tmp_train_file);
    if  ~exist(tmp_train_file) | fileInfo.bytes < 1024
        tt_list(counter) = [];
        labels(counter) = [];
        counter = counter - 1;
    else
        counter = counter + 1;
    end  
end

switch t_type
    case 'train'
        %extract DTF data and set labels
        %try
        %	matlabpool close;
        %catch exception
        %end
        %matlabpool open 4
        Trajectory=[];
        HOG=[];
        HOF=[];
        MBHx=[];
        MBHy=[];
        parfor i=1:length(tt_list)
            
            pathOfIdtBin=fullfile(params.dtf_dir,tt_list{i}(1:length(tt_list{i}))); 
            [newTrajectory,newHOG,newHOF,newMBHx,newMBHy]=extract_dtf_feats(params, pathOfIdtBin,params.DTF_subsample_num,'codebook');
            
            Trajectory=[Trajectory newTrajectory];
            HOG=[HOG newHOG];
            HOF=[HOF newHOF];
            MBHx=[MBHx newMBHx];
            MBHy=[MBHy newMBHy];
        end
        %matlabpool close
        
        X{1}=Trajectory;
        clear Trajectory;
        X{2}=HOG;
        clear HOG;
        X{3}=HOF;
        clear HOF;
        X{4}=MBHx;
        clear MBHx;
        X{5}=MBHy;
        clear MBHy;
    case 'test'
        % Do nothing for test videos
    otherwise
        error('Unknown file pattern!');
end

file_list=tt_list;

end
