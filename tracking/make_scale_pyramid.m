% -----------------------------------------------------------------------------------------------------
function pyramid = make_scale_pyramid(im, targetPosition, in_side_scaled, out_side,       avgChans, stats, p)
%        x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
%MAKE_SCALE_PYRAMID
%   computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
%   and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016

% 这个函数槽点巨多，首先两次调用 get_subwindow_tracking 函数其实可以一次解决，其次，两次对图像进行缩放而且缩放大小相近，误差极大！
% 总之其功能就是在 im 图像中，以 targetPosition 为中心，按照 in_side_scaled （向量）为大小进行正方形裁剪，然后压缩到 out_side 的大小（255*255）
% 用 avgChans 做 padding ，可能减去 stats 中的平均值。
% -----------------------------------------------------------------------------------------------------
	
	% in_side_scaled 	缩放后正方形图像在原系的边长。
	% out_side			输出正方形的边长，此处为 255 .
    in_side_scaled = round(in_side_scaled);	
    pyramid = gpuArray(zeros(out_side, out_side, 3, p.numScale, 'single'));
    max_target_side = in_side_scaled(end);
    min_target_side = in_side_scaled(1);
    beta = out_side / min_target_side;
    % size_in_search_area = beta * size_in_image
    % e.g. out_side = beta * min_target_side
    search_side = round(beta * max_target_side);
    [search_region, ~] = get_subwindow_tracking(im, targetPosition, [search_side search_side], [max_target_side max_target_side], avgChans);
    if p.subMean
        search_region = bsxfun(@minus, search_region, reshape(stats.x.rgbMean, [1 1 3]));
    end
    assert(round(beta * min_target_side)==out_side);

    for s = 1:p.numScale
        target_side = round(beta * in_side_scaled(s));
        pyramid(:,:,:,s) = get_subwindow_tracking(search_region, (1+search_side*[1 1])/2, [out_side out_side], target_side*[1 1], avgChans);
    end
end