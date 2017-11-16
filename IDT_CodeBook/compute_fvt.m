function fvt = compute_fvt(params, pca_coeff, gmm, pos_files,type)

fvt=[];
        for i=1:length(pos_files)
            % pathOfIdtBin=fullfile(params.dtf_dir,regexprep(pos_files{i},'.avi',''));
%              disp(pathOfIdtBin);
            switch type
                case 'training'
                    pathOfIdtBin=fullfile(params.dtf_dir,pos_files{i}(1:length(pos_files{i})));
                    %fileInfo=dir([pathOfIdtBin,'_iDT_Features.bin']);
                    %fileSize=fileInfo.bytes();
                    %if
                        
                    feature=import_idt([pathOfIdtBin,'_iDT_Features.bin'],15);
                   % feature=import_idt_hog3d([pathOfIdtBin,'_iDT_Features_depth.bin'],15);
                case 'codebook'
                    pathOfIdtBin=fullfile(params.dtf_dir,pos_files{i}(1:length(pos_files{i})));
                    feature=import_idt([pathOfIdtBin,'_iDT_Features.bin'],15);
            end
            
           
            Trajectory = pca_coeff{1} * feature.tra_shape; 
            fv_trajectory=vl_fisher(Trajectory, gmm{1}.means, gmm{1}.covariances, gmm{1}.priors);
            HOG = pca_coeff{2} * feature.hog;
            fv_hog=vl_fisher(HOG, gmm{2}.means, gmm{2}.covariances, gmm{2}.priors);
            HOF = pca_coeff{3} * feature.hof;
            fv_hof=vl_fisher(HOF, gmm{3}.means, gmm{3}.covariances, gmm{3}.priors);
            MBHx = pca_coeff{4} * feature.mbhx;
            fv_mbhx=vl_fisher(MBHx, gmm{4}.means, gmm{4}.covariances, gmm{4}.priors);
            MBHy = pca_coeff{5} * feature.mbhy;
            fv_mbhy=vl_fisher(MBHy, gmm{5}.means, gmm{5}.covariances, gmm{5}.priors);
            fv=[fv_hog;fv_hof;fv_mbhx;fv_mbhy];
            fvt=[fvt fv];            
        end
        
        % power normalization
        fvt = sign(fvt) .* sqrt(abs(fvt));
        % L2 normalization
        fvt = double(yael_fvecs_normalize(single(fvt)));
end