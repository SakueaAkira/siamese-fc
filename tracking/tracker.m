% -------------------------------------------------------------------------------------------------
function bboxes = tracker(varargin)
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
%   本函数中涉及到 卷积网络中图像的坐标系 和 实际图像的坐标系 ，以及他们之间的坐标变换，简称两个坐标系为 网系 和 原系 。
% -------------------------------------------------------------------------------------------------

%% -------------------------------------- 超参数 ----------------------------------------------------
   
    % SiamFC-3S 的超参
    p.numScale = 3;					% 缩放级数（在用 z 搜索 x 时 x 的变化分几级）
    p.scaleStep = 1.0375;			% 缩放步长（缩放 x 时按照这个参数的 n 次方缩放）
    p.scalePenalty = 0.9745;
    p.scaleLR = 0.59; 				% damping factor for scale update
    p.responseUp = 16; 				% 上采样倍数。将全卷积网络的 17*17 激活图进行上采样（插值放大），提高定位精度。
    p.windowing = 'cosine'; 		% 使用汉宁窗防止位移过大
    p.wInfluence = 0.176; 			% windowing influence (in convex sum) 0.176
    p.net = '2016-08-17.net.mat'; 
    
    % 5S的超参 
%     p.numScale = 5;
%     p.scaleStep = 1.0255;
%     p.scalePenalty = 0.962;  % penalizes the change of scale
%     p.scaleLR = 0.34;
%     p.responseUp = 16; % response upsampling factor (purpose is to account for stride, but they dont have to be equal)
%     p.windowing = 'cosine';
%     p.wInfluence = 0.168;
%     p.net = '2016-08-17.net.mat';
    
%% -----------------------------------------调试选项参数-----------------------------------------------------    

    % execution, visualization, benchmark
    p.video = 'vot15_bag';
    p.visualization = false;
    p.gpus = 1;
    p.bbox_output = false;
    p.fout = -1;
	
    % 网络的结构参数，训练时必须为常量。
    p.exemplarSize = 127;  % input z size
    p.instanceSize = 255;  % input x size (search region)
    p.scoreSize = 17;
    p.totalStride = 8;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;
	
	
    % 网络名称参数
    p.prefix_z = 'a_'; % used to identify the layers of the exemplar
    p.prefix_x = 'b_'; % used to identify the layers of the instance
    p.prefix_join = 'xcorr';
    p.prefix_adj = 'adjust';
    p.id_feat_z = 'a_feat';
    p.id_score = 'score';
	
    % 将默认参数覆盖。
    p = vl_argparse(p, varargin);
%% -------------------------------------------------------------------------------------------------

    % Get environment-specific default paths.
    p = env_paths_tracking(p);
    % Load ImageNet Video statistics
    if exist(p.stats_path,'file')
        stats = load(p.stats_path);
    else
        warning('No stats found at %s', p.stats_path);
        stats = [];
    end
	
%% ------------------------------------网络构建------------------------------------------------------

    % 原网络文件中保存了双生网络的数据。分别构造两个网络实例。
	% 样本特征提取网络，从第一帧中。
    net_z = load_pretrained([p.net_base_path p.net], p.gpus);
    remove_layers_from_prefix(net_z, p.prefix_x);
    remove_layers_from_prefix(net_z, p.prefix_join);
    remove_layers_from_prefix(net_z, p.prefix_adj);
	
	% 搜索网络。保留了最后的卷积层。
    net_x = load_pretrained([p.net_base_path p.net], []);
    remove_layers_from_prefix(net_x, p.prefix_z);
	
	%取得样本输出层和score层的ID(ID 实际上就是 layer 在数组中第几个)
	zFeatId = net_z.getVarIndex(p.id_feat_z);
    scoreId = net_x.getVarIndex(p.id_score);
	
%% -----------------------------------载入视频----------------------------------------------------------	
	
	% 载入了全部视频和第一帧的 Bound Box 信息，注意 targetPosition 和 targetSize 之后还要使用，现在作为初始信息。
	
    [imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);			%p.video is like 'vot15_xxx'
																								%p.seq_base_path is directory of video sequences.
																								%imgFiles is loaded video sequences file handle by vl.
																								%targetPosition is vector [cy cx] depicts centre of bound box.
																								%targetSize is vector [h w]
    nImgs = numel(imgFiles);
    startFrame = 1;

    % 取得第一帧图像，存储于GPU数组，开始GPU计算。
    im = gpuArray(single(imgFiles{startFrame}));
    % 如果是灰度图，将其重复三个通道。
	if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    % Init visualization
    videoPlayer = [];
    if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
        videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
    end
	
    % 在GPU中计算第一帧三个通道的平均值，并将结果用gather函数搬回CPU中。
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

	% 用一个稍大一些的正方形将ground truth包括的部分框起来，正方形的边长为 s_z ，附上几何画板文件展示这个框。
    wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
    hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
    s_z = sqrt(wc_z*hc_z);
    scale_z = p.exemplarSize / s_z;            % scale_z 输入网络的正方形图像与原始正方形图像大小之比，实际上就是网络与原始图像的比率。
	
    % 得到裁剪为输入大小（127*127) 的样本的图像，如果正方形越界，则用通道平均值 padding 越界部分。
	% 这个函数问题不少。
    [z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
	
	% 视情况减去 status 中图像的平均值。
    if p.subMean
        z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
    end
	
    d_search = (p.instanceSize - p.exemplarSize)/2;			% 网系中搜索框定位点范围的边长的一半。（实际上搜索框定位点的范围是一个正方形，这里求他在网系中的边长）
    pad = d_search / scale_z;								% 这里求原系中搜索范围相对于搜索框的四周 padding 大小。
    s_x = s_z + 2*pad;										% 求原坐标系中搜索范围的边长。
    % arbitrary scale saturation
    min_s_x = 0.2*s_x;										% 最大和最小搜索范围大小。（看来是要examlper不变然后缩放search reigon）
    max_s_x = 5*s_x;

    switch p.windowing
        case 'cosine'
			% 生成对于 (17*16)^2 Score 图的余弦窗（Hanning 窗）
            window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
        case 'uniform'
			% 不生成窗函数。
            window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    end
	
	% 将 window 正则化，使其和为 1 .
    window = window / sum(window(:));% window(:) 是 window 平铺于一维的结果。
	% 后面那串东西			 (		(ceil(p.numScale/2)-p.numScale)		:		floor(p.numScale/2)		)
	% 当 numScale 为 3 时生成      [-1 0 1]					[0.9638 1 1.0375]
	% 当 numScale 为 5 时生成   [-2 -1 0 1 2]
    scales = (p.scaleStep .^ (		(ceil(p.numScale/2)-p.numScale)		:		floor(p.numScale/2)		));
    % evaluate the offline-trained network for exemplar z features
	% 将 exemplar 传入网络 z 进行计算，得到其 feature 。
    net_z.eval({'exemplar', z_crop});
    z_features = net_z.vars(zFeatId).value;						% size: 6*6*128
    z_features = repmat(z_features, [1 1 1 p.numScale]);		% 在第四维上复制了三份

    bboxes = zeros(nImgs, 4);
    % start tracking
    tic;
    for i = startFrame:nImgs
	
        if i>startFrame
            % load new frame on GPU
            im = gpuArray(single(imgFiles{i}));
   			% 单通道灰度图就重复三遍
    		if(size(im, 3)==1)
        		im = repmat(im, [1 1 3]);
    		end
			% 计算缩放参数
            scaledInstance = s_x .* scales;
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
            % 在先前的目标位置裁剪一个正方形搜索区域，按照缩放比例形成缩放金字塔。
            x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            % evaluate the offline-trained network for exemplar x features
            [newTargetPosition, newScale] = tracker_eval(net_x, round(s_x), scoreId, z_features, x_crops, targetPosition, window, p);
            targetPosition = gather(newTargetPosition);
            % scale damping and saturation
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        else
            % 第一帧不 track ，直接用 Ground Truth 做第一帧结果。
        end
		
		
		% targetPosition([2,1]) 这种写法将 targetPosition 变量互换。 原来存储形式为 cy, cx 现在反过来。
		% rectPosition 是 Bound Box 左上角坐标。
        rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
        % output bbox in the original frame coordinates
        oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
        oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
        bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];

        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(im/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 3, 'EdgeColor', 'y');
                drawnow
                fprintf('Frame %d\n', startFrame+i);
            else
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 3, 'Color', 'cyan');
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
            end
        end

        if p.bbox_output
            fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', bboxes(i, :));
        end

    end

    bboxes = bboxes(startFrame : i, :);

end
