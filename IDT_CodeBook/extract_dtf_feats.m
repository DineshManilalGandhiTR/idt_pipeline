function [ Trajectory,HOG,HOF,MBHx,MBHy ] = extract_dtf_feats( params, path, num_feats,type )
%EXTRACT_DTF_FEATS extract DTF features.
%   The first 10 elements for each line in dtf_file are information about the trajectory.
%   The trajectory info(default 30 dimensions) should also be discarded.
%	
%   Subsampling:
%       randomly choose 100 descriptors from each video clip(dtf file)
%		To use all the DTF fatures, set num_feats to a negative number.
%
%feature=import_idt('iDT_Features.bin',15);

% HOG=zeros(params.feat_len_map('HOG'),1);
% HOF=zeros(params.feat_len_map('HOF'),1);
% MBHx=zeros(params.feat_len_map('MBHx'),1);
% MBHy=zeros(params.feat_len_map('MBHy'),1);

Trajectory=zeros(30,1);
HOG=zeros(96,1);
HOF=zeros(108,1);
MBHx=zeros(96,1);
MBHy=zeros(96,1);

% if ~exist(dtf_file,'file')
% 	warning('File %s does not exist! Skip now...',dtf_file);
% 	return;
% else
% 	tmpfile=dir(dtf_file);
% 	if tmpfile.bytes < 1024
% 		warning('File %s is too small! Skip now...',dtf_file);
% 		return;
% 	end
% end
  
hog_range=params.feat_start:params.feat_start+params.feat_len_map('HOG')-1;
hof_range=hog_range(end)+1:hog_range(end)+params.feat_len_map('HOF');
mbhx_range=hof_range(end)+1:hof_range(end)+params.feat_len_map('MBHx');
mbhy_range=mbhx_range(end)+1:mbhx_range(end)+params.feat_len_map('MBHy');

disp(path);
switch type
    case 'training'
        feature=import_idt([path,'_iDT_Feature.bin'],15);
        %feature=import_idt(path,15);
    case 'codebook'
        feature=import_idt([path,'_iDT_Features.bin'],15);
end

sizenum=size(feature.hog);sizenum=sizenum(2);
% sizehof=size(feature.hof);sizehof=sizehof(2);
% sizembhx=size(feature.mbhx);sizembhx=sizembhx(2);
% sizembhy=size(feature.mbhy);sizembhy=sizembhy(2);

if num_feats<0 % To use all the DTF fatures, set num_feats to a negative number
% 	num_feats=size(x,1);
    num_feats=sizenum;
end

if sizenum<=num_feats
	idx=1:sizenum; % randomly subsampling
else
	idx=randperm(sizenum,num_feats); % randomly subsampling
	%idx=floor(linspace(1,size(x,1),num_feats)); % linearly subsampling
end

Trajectory=feature.tra_shape(:,idx);
HOG=feature.hog(:,idx);
HOF=feature.hof(:,idx);
MBHx=feature.mbhx(:,idx);
MBHy=feature.mbhy(:,idx);

end

