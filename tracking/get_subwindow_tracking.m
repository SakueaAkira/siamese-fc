% --------------------------------------------------------------------------------------------------------
function [im_patch, im_patch_original] = get_subwindow_tracking(im, pos,             model_sz,                       original_sz,             avg_chans)
%        [z_crop,   ~] =                 get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
% GET_SUBWINDOW_TRACKING Obtain image sub-window, padding with avg channel if area goes outside of border
% 按照给定的坐标将以坐标为中心，边长 original_sz 的正方形图像裁剪出来为 im_patch_original ，并且制作一个边长为 model_sz 的缩放版本为 im_patch。
% 如果给定的正方形大小超出了边界，则在空白位置上补充通道的平均值。
% 有几个问题：
%	1. 按照返回参数来看，可以直接将目标框弄出来然后再在旁边padding，但是这个函数是先将原图像padding扩大后再裁剪，有些拖慢算法。
%	2. 函数并未传入 ground truth 的任何信息，这样会将 ground truth 旁边的无关图像也包含进去，而不是将其 padding，这样真的好吗？
%	3. 为何只有正方形边框在左边界或上边界越界时才将其拉回视野内？ 而且这样会导致左上越界时 padding 事实上无效，因为回传的图像是拉回视野后的正方形。
% -------------------------------------------------------------------------------------------------
    if isempty(original_sz)
        original_sz = model_sz;
    end
    sz = original_sz;
    im_sz = size(im);
    %make sure the size is not too small
	% 确保每一维度上长度都大于2.
    assert(all(im_sz(1:2) > 2));
    c = (sz+1) / 2;

    % check out-of-bounds coordinates, and set them to black
	% 计算包围正方形的四点坐标，可能出现负值或越界值。
    context_xmin = round(pos(2) - c(2)); % floor(pos(2) - sz(2)/2);
    context_xmax = context_xmin + sz(2) - 1;
    context_ymin = round(pos(1) - c(1)); % floor(pos(1) - sz(1)/2);
    context_ymax = context_ymin + sz(1) - 1;
	
	% 计算外包正方形超出视野边界的距离，如果正方形未越边则值为0.
    left_pad = max(0, 1-context_xmin);
    top_pad = max(0, 1-context_ymin);
    right_pad = max(0, context_xmax - im_sz(2));
    bottom_pad = max(0, context_ymax - im_sz(1));

	% 如果超出了左上边界，则将外包正方形移回视野内（为什么？）。
    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

	% 如果外包正方形有溢出的话，将第一帧图像用各通道平均值扩充。
    if top_pad || left_pad
        R = padarray(im(:,:,1), [top_pad left_pad], avg_chans(1), 'pre');
        G = padarray(im(:,:,2), [top_pad left_pad], avg_chans(2), 'pre');
        B = padarray(im(:,:,3), [top_pad left_pad], avg_chans(3), 'pre');
        im = cat(3, R, G, B);
    end
    if bottom_pad || right_pad
        R = padarray(im(:,:,1), [bottom_pad right_pad], avg_chans(1), 'post');
        G = padarray(im(:,:,2), [bottom_pad right_pad], avg_chans(2), 'post');
        B = padarray(im(:,:,3), [bottom_pad right_pad], avg_chans(3), 'post');
        im = cat(3, R, G, B);
    end

	% 按照外包正方形裁剪图像为 im_patch_original
    xs = context_xmin : context_xmax;
    ys = context_ymin : context_ymax;
	im_patch_original = im(ys, xs, :);
	
	% 将外包正方形图像裁剪为输入大小
    if ~isequal(model_sz, original_sz)
        im_patch = imresize(im_patch_original, model_sz(1)/original_sz(1));
        % Strangely, sometimes model_sz/original_sz doesn't output model_sz and output of rescaling has to be forced to model_sz
        % (gpu version of imresize does not allow array as model size, only ratio
		% 有时缩放函数的第二个参数是目标图像的大小而不是比率，这取决于GPU版本。
        if size(im_patch,1)~=model_sz(1)
            % WARNING: camera ready used mexResize and not imresize, this could cause minor differences in results
%             im_patch = gpuArray(mexResize(gather(im_patch_original), model_sz, 'auto'));
			% （好像是）相机的图像缩放函数不同，是 mexResize 函数。
            im_patch = gpuArray(imresize(gather(im_patch_original), model_sz));
        end
    else
        im_patch = im_patch_original;
    end
end
