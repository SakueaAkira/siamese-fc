% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale] = tracker_eval(net_x, s_x, scoreId, z_features, x_crops, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    % forward pass, using the pyramid of scaled crops as a "batch"
	% 输入 z 的 feature 和 x 的各缩放裁剪结果进行激活。
    net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});
    responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
    % 选择峰值最高的响应图
    if p.numScale>1
		% currentScaleID 假设 scaleNum=5 , [-2 -1 0 1 2] 无缩放时对应的 0 序号为 3 ，即 ceil（5/2）。
        currentScaleID = ceil(p.numScale/2);
        bestScale = currentScaleID;
        bestPeak = -Inf;
		
		
        for s=1:p.numScale
			% 上采样提升精度
            if p.responseUp > 1
				% 双三次插值，将图像扩大 responseUp 倍。
                responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
            else
                responseMapsUP(:,:,s) = responseMaps(:,:,s);
            end
            thisResponse = responseMapsUP(:,:,s);
            % 如果不是无缩放的状态，就乘以一个降权系数。
            if s~=currentScaleID, thisResponse = thisResponse * p.scalePenalty; end
            thisPeak = max(thisResponse(:));
            if thisPeak > bestPeak, bestPeak = thisPeak; bestScale = s; end
        end
		% 最终的响应图选择响应最大的那副图。
        responseMap = responseMapsUP(:,:,bestScale);
    else
        responseMap = responseMapsUP;
        bestScale = 1;
    end
    % 响应图归一化，使其和为 1.
    responseMap = responseMap - min(responseMap(:));
    responseMap = responseMap / sum(responseMap(:));
    % 应用位移降权窗。降权窗和响应图按比例 wInfluence 混合。wInfluence 为 1 时响应图就是降权窗，新位置停留在上一帧。 
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
	% 找到最大值坐标。
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    % 将上采样网系转到原系。上采样网系大小为 272*272
    % 上采样网系中的位移向量
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
	% 将 272*272 映射回 instance 中，instance中新的目标位置只可能存在于与 255*255 同心的 127*127 正方形中。
	% 因此实际上是将 272*272 -> 127*127 而不是 272*272 -> 255*255
	% 现在 totalStride=8 ，这个公式有助于理解： (255-127)/8+1=17
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % 最后将其映射回原系。似乎没有考虑缩放。
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % 通过位移向量计算新位置。
    newTargetPosition = targetPosition + disp_instanceFrame;
end

% 这个函数似乎并未考虑上采样。
function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
